#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Provision / inspect / tear down the rented box a session runs on (Prime Intellect).

The sibling of `scripts/runpod_session.py`, same contract and same subcommands, for
a second venue. Written because RunPod's 8-GPU pool went dry (2026-08-19: 36
consecutive `avail` misses) while Prime Intellect had 8xA100-80GB SXM4 in stock.

    uv run --script scripts/prime_session.py status
    uv run --script scripts/prime_session.py avail             # live 8-GPU stock
    uv run --script scripts/prime_session.py up --dry-run      # the plan, free
    uv run --script scripts/prime_session.py up --max-hours 4
    uv run --script scripts/prime_session.py guard             # the cost kill-switch
    uv run --script scripts/prime_session.py ssh [--exec]
    uv run --script scripts/prime_session.py down
    uv run --script scripts/prime_session.py verify [--strict]  # is anything billing?

Prime Intellect is an *exchange*: `provider.type` selects an upstream (lambdalabs,
vultr, hyperstack, runpod, ...) and every offer is priced and stocked separately.
So placement here picks a (provider, data center, socket) triple, not just a data
center. Plain REST + a bearer token, so this needs no SDK -- stdlib only.

Deliberate, load-bearing choices (docs/decisions.md §9 and §20):

- **The socket is pinned, and defaults to SXM4.** The 8xA100 anchor (§14, 269.9
  TFLOP/s measured, NVLink in `topo.txt`) is an SXM4 box. Prime Intellect lists
  A100_80GB in both SXM4 and PCIe, at similar prices, and a PCIe box would produce
  a plausible-looking number that is not comparable to the anchor. Renting the
  wrong socket is this venue's version of mounting a volume over /workspace: it
  fakes the comparison rather than failing it. `--socket any` is deliberate opt-out.
- **Image parity is refused by default, not silently skipped.** PI's `image` field
  is a fixed enum of *their* environments; a private image is reachable only as
  `image=custom_template` + a `customTemplateId` registered in the console. With
  no template id the pod boots THEIR ubuntu, no distrain code, and any result from
  it is unreportable. `up` therefore refuses without `--template-id` unless
  `--allow-stock-image` says so out loud.
- **Teardown on exception.** Anything that fails after the pod exists terminates
  it before exiting, so a half-provisioned box cannot bill quietly.
- **`up` refuses to start what the balance cannot finish** -- credit exhaustion
  terminated a nearly-converged run once (2026-08-16 session log §6a); `guard`
  then enforces the wall-clock ceiling and watches the balance for the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_FILE = REPO_ROOT / "out" / "prime" / "session.json"

BASE_URL = "https://api.primeintellect.ai"
GPU_TYPE = "A100_80GB"          # PI's enum spelling; see GPUType in their OpenAPI
GPU_COUNT = 8
SOCKET = "SXM4"                 # NVLink. The anchor's box -- see the module docstring.
DISK_GB = 200                   # FineWeb shards (~10.5 GiB) + checkpoints + room
IMAGE_REPO = "ghcr.io/adamdivak/distrain"

# Terminal states: a pod in one of these is not billing and cannot be revived.
DEAD_STATUSES = {"TERMINATED", "DELETING", "ERROR"}
LIVE_STATUSES = {"PROVISIONING", "PENDING", "ACTIVE"}


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #

def load_api_key(env_file: Path = REPO_ROOT / ".env") -> str:
    """PRIME_API_KEY from the environment, else from the gitignored .env, else
    from ~/.prime/config.json (where the `prime` CLI and SkyPilot keep it)."""
    key = os.environ.get("PRIME_API_KEY", "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "PRIME_API_KEY":
                key = value.strip().strip("'\"")
                break
    if not key:
        cfg = Path.home() / ".prime" / "config.json"
        if cfg.exists():
            try:
                key = str(json.loads(cfg.read_text()).get("api_key") or "").strip()
            except (json.JSONDecodeError, OSError):
                key = ""
    if not key:
        sys.exit(f"no PRIME_API_KEY in the environment, {env_file}, or ~/.prime/config.json")
    return key


def load_team_id(env_file: Path = REPO_ROOT / ".env") -> str | None:
    """PRIME_TEAM_ID, else `.env`, else the `prime` CLI's own `team_id`.

    Console top-ups land on a team wallet, and a bare key sees only the personal
    one -- so the team is part of the credentials, not an option.
    """
    team = os.environ.get("PRIME_TEAM_ID", "").strip()
    if not team and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "PRIME_TEAM_ID":
                team = value.strip().strip("'\"")
                break
    if not team:
        cfg = Path.home() / ".prime" / "config.json"
        if cfg.exists():
            try:
                team = str(json.loads(cfg.read_text()).get("team_id") or "").strip()
            except (json.JSONDecodeError, OSError):
                team = ""
    return team or None


# --------------------------------------------------------------------------- #
# API surface -- one seam, so the tests can drive the whole script with a fake
# --------------------------------------------------------------------------- #

class PrimeAPI:
    """Thin wrapper over Prime Intellect's REST API (stdlib urllib, no SDK).

    Endpoint names come from their published OpenAPI document rather than the
    prose docs, which lag it.
    """

    def __init__(self, api_key: str, base_url: str = BASE_URL, timeout: int = 60,
                 team_id: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # A key resolves to the *personal* wallet unless every call carries the
        # team. Money added in the console lands on a team wallet by default, so
        # without this the balance reads $0.00 and `up` refuses a funded account.
        self.team_id = team_id

    def _request(self, method: str, path: str, *, params: dict | None = None,
                 body: dict | None = None) -> dict | list:
        url = f"{self.base_url}{path}"
        params = dict(params or {})
        if self.team_id:
            params.setdefault("teamId", self.team_id)
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc

    # -- account ----------------------------------------------------------- #

    def wallet(self) -> dict:
        """Balance. Note PI bills a *wallet*; `balance_usd` is what `up` budgets on."""
        return self._request("GET", "/api/v1/billing/wallet")

    def whoami(self) -> dict:
        return self._request("GET", "/api/v1/user/whoami")

    def teams(self) -> list[dict]:
        got = self._request("GET", "/api/v1/user/teams")
        return (got or {}).get("data", []) if isinstance(got, dict) else []

    def registry_credentials(self) -> list[dict]:
        """Private-registry credentials registered in the console. Needed for the
        GHCR image; there is no API to create one, only to list."""
        got = self._request("GET", "/api/v1/template/registry-credentials")
        return (got or {}).get("credentials", []) if isinstance(got, dict) else []

    # -- availability ------------------------------------------------------ #

    def availability(self, gpu_type: str | None = None,
                     gpu_count: int | None = None) -> dict:
        params: dict = {}
        if gpu_type:
            params["gpu_type"] = gpu_type
        if gpu_count:
            params["gpu_count"] = gpu_count
        got = self._request("GET", "/api/v1/availability/", params=params)
        return got if isinstance(got, dict) else {}

    # -- pods -------------------------------------------------------------- #

    def pods(self) -> list[dict]:
        got = self._request("GET", "/api/v1/pods/")
        return (got or {}).get("data", []) if isinstance(got, dict) else []

    def pod(self, pod_id: str) -> dict | None:
        try:
            got = self._request("GET", f"/api/v1/pods/{pod_id}")
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        return got if isinstance(got, dict) else None

    def create_pod(self, payload: dict) -> dict:
        # Pod creation takes the team in the *body*, not as a query param -- and
        # it decides which wallet the pod bills to, so it must not be omitted.
        if self.team_id:
            payload = {**payload, "team": {"teamId": self.team_id}}
        got = self._request("POST", "/api/v1/pods/", body=payload)
        return got if isinstance(got, dict) else {}

    def terminate_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/api/v1/pods/{pod_id}")

    # -- ssh keys ---------------------------------------------------------- #

    def ssh_keys(self) -> list[dict]:
        got = self._request("GET", "/api/v1/ssh_keys/")
        if isinstance(got, dict):
            return got.get("data", got.get("ssh_keys", []))
        return got if isinstance(got, list) else []

    def create_ssh_key(self, name: str, public_key: str) -> dict:
        got = self._request("POST", "/api/v1/ssh_keys/",
                            body={"name": name, "publicKey": public_key})
        return got if isinstance(got, dict) else {}


# --------------------------------------------------------------------------- #
# pure helpers -- the arithmetic and parsing, kept testable without an API
# --------------------------------------------------------------------------- #

def offer_price(offer: dict) -> float | None:
    prices = offer.get("prices") or {}
    price = prices.get("onDemand")
    return float(price) if price is not None else None


def in_stock(offer: dict) -> bool:
    """PI reports 'Available' / 'Low' / 'Unavailable'. Low stock still rents."""
    status = str(offer.get("stockStatus") or "").strip().lower()
    return status not in ("unavailable", "none", "out_of_stock", "")


def matching_offers(availability: dict, gpu_type: str, gpu_count: int,
                    socket: str | None = None,
                    provider: str | None = None) -> list[dict]:
    """In-stock offers for exactly this shape, cheapest first.

    `socket` defaults to SXM4 at the call sites: A100_80GB is listed in both SXM4
    and PCIe and the two are not interchangeable for a scaling comparison.
    """
    offers = []
    for listed_gpu, entries in (availability or {}).items():
        if listed_gpu != gpu_type:
            continue
        for offer in entries or []:
            if int(offer.get("gpuCount") or 0) != gpu_count:
                continue
            if socket and socket != "any" and str(offer.get("socket") or "") != socket:
                continue
            if provider and str(offer.get("provider") or "") != provider:
                continue
            if not in_stock(offer):
                continue
            if offer_price(offer) is None:
                continue
            offers.append(offer)
    return sorted(offers, key=lambda o: offer_price(o) or float("inf"))


def parse_ssh_connection(ssh_connection) -> tuple[str | None, str | None, int]:
    """(user, host, port) from PI's `sshConnection`, which varies by upstream.

    Seen shapes: "ubuntu@1.2.3.4", "ubuntu@1.2.3.4 -p 2222",
    "ubuntu@1.2.3.4:2222", and either of those inside a one-element list.
    """
    tokens: list[str] = []
    if isinstance(ssh_connection, str):
        tokens = ssh_connection.split()
    elif isinstance(ssh_connection, list):
        for item in ssh_connection:
            tokens.extend(str(item).split())
    else:
        return None, None, 22

    user = host = None
    port = 22
    for index, token in enumerate(tokens):
        if token in ("-p", "--port") and index + 1 < len(tokens):
            try:
                port = int(tokens[index + 1])
            except ValueError:
                pass
        elif "@" in token:
            user, _, hostpart = token.partition("@")
            if ":" in hostpart:
                hostpart, _, portpart = hostpart.partition(":")
                try:
                    port = int(portpart)
                except ValueError:
                    pass
            host = hostpart
    return user, host, port


def ssh_target(pod: dict) -> tuple[str, str, int] | None:
    """(user, ip, port) once the pod is reachable, else None."""
    user, host, port = parse_ssh_connection(pod.get("sshConnection"))
    host = host or pod.get("ip")
    if not host:
        return None
    return (user or "root"), host, port


def ssh_command(user: str, ip: str, port: int) -> str:
    return f"ssh -p {port} {user}@{ip}"


def runway_hours(balance: float, cost_per_hr: float) -> float:
    """Hours of spend the remaining credit covers. inf when nothing is running."""
    return float("inf") if cost_per_hr <= 0 else balance / cost_per_hr


def budget_verdict(balance: float, price_per_hr: float, max_hours: float) -> str | None:
    """Refusal message when the balance cannot fund the whole ceiling, else None."""
    need = price_per_hr * max_hours
    if balance < need:
        return (
            f"balance ${balance:.2f} cannot fund {max_hours:g} h at ${price_per_hr:.2f}/h "
            f"(${need:.2f}). Top up, or lower --max-hours to "
            f"{runway_hours(balance, price_per_hr):.1f} h. A pod that outlives its credit "
            "is terminated with its disk, so the spend buys nothing."
        )
    return None


def deadline_iso(started: datetime, max_hours: float) -> str:
    return (started + timedelta(hours=max_hours)).replace(microsecond=0).isoformat()


def read_pubkey(path: Path | None) -> str:
    candidates = [path] if path else [
        Path.home() / ".ssh" / "id_ed25519.pub",
        Path.home() / ".ssh" / "id_rsa.pub",
    ]
    for candidate in candidates:
        if candidate and candidate.expanduser().exists():
            return candidate.expanduser().read_text().strip()
    return ""


def git_image_tag() -> tuple[str, bool]:
    """(short sha, tree_is_clean). The image tag is the provenance of every number
    a session produces, so a dirty tree is worth a warning."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    return sha, not dirty


def find_pod(pods: list[dict], name: str) -> dict | None:
    for pod in pods:
        if pod.get("name") == name and str(pod.get("status")) not in DEAD_STATUSES:
            return pod
    return None


# --------------------------------------------------------------------------- #
# session file
# --------------------------------------------------------------------------- #

def write_session(data: dict, path: Path = SESSION_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_session(path: Path = SESSION_FILE) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_status(api: PrimeAPI, args: argparse.Namespace) -> int:
    wallet = api.wallet()
    balance = float(wallet.get("balance_usd") or 0.0)
    scope = f"team {wallet.get('team_id')}" if wallet.get("team_id") else "personal"
    print(f"balance    ${balance:.2f}  ({scope} wallet {wallet.get('wallet_id')})")
    if balance == 0.0 and not api.team_id:
        teams = api.teams()
        if teams:
            print("           this is the personal wallet and it is empty. Funds added in "
                  "the console land on a team wallet -- try --team-id:")
            for team in teams:
                print(f"             {team.get('teamId')}  {team.get('name')}")

    pods = api.pods()
    live = [p for p in pods if str(p.get("status")) in LIVE_STATUSES]
    print(f"pods       {len(pods)} total, {len(live)} live")
    for pod in pods:
        price = float(pod.get("priceHr") or 0)
        print(f"  {pod.get('id')}  {pod.get('name')}  {pod.get('status')}  "
              f"{pod.get('gpuCount')}x {pod.get('gpuName')} {pod.get('socket')}  "
              f"${price:.2f}/h  {pod.get('providerType')}")

    creds = api.registry_credentials()
    print(f"registry   {len(creds)} credential(s): "
          f"{', '.join(str(c.get('name') or c.get('id')) for c in creds) or '(none)'}")
    if not creds:
        print("           WARNING: no private-registry credential. The GHCR image cannot "
              "be pulled, so a custom template cannot be built. Add one in the console.")
    return 0


def cmd_verify(api: PrimeAPI, args: argparse.Namespace) -> int:
    """Is anything billing? The post-session check that nothing leaked."""
    pods = api.pods()
    live = [p for p in pods if str(p.get("status")) in LIVE_STATUSES]
    wallet = api.wallet()
    balance = float(wallet.get("balance_usd") or 0.0)

    burn = sum(float(p.get("priceHr") or 0) for p in live)
    print(f"balance ${balance:.2f}, {len(live)} live pod(s), ${burn:.2f}/h burning")
    for pod in live:
        print(f"  LIVE {pod.get('id')} {pod.get('name')} {pod.get('status')} "
              f"${float(pod.get('priceHr') or 0):.2f}/h")
    if live and args.strict:
        print("\nstrict: something is still billing.")
        return 1
    if not live:
        print("nothing is billing.")
    return 0


def cmd_avail(api: PrimeAPI, args: argparse.Namespace) -> int:
    """Live stock for the wanted shape. Exit 0 when something is rentable, so
    `watch_capacity.sh` can poll this the same way it polls RunPod."""
    availability = api.availability(gpu_type=args.gpu_type, gpu_count=args.gpu_count)
    offers = matching_offers(availability, args.gpu_type, args.gpu_count,
                             socket=args.socket, provider=args.provider)

    label = f"{args.gpu_count}x {args.gpu_type}"
    if args.socket and args.socket != "any":
        label += f" {args.socket}"
    print(f"{label}")
    if not offers:
        others = matching_offers(availability, args.gpu_type, args.gpu_count, socket="any")
        print("  no in-stock offers for that shape.")
        if others:
            sockets = sorted({str(o.get('socket')) for o in others})
            print(f"  (in stock at other sockets: {', '.join(sockets)} -- "
                  "not interchangeable for the anchor comparison)")
        return 1

    for offer in offers:
        print(f"  {offer.get('provider'):<14} {offer.get('socket'):<6} "
              f"{offer.get('dataCenter')!s:<16} {offer.get('stockStatus')!s:<10} "
              f"${offer_price(offer):.2f}/h")
    return 0


def _ensure_ssh_key(api: PrimeAPI, pubkey: str, name: str) -> str | None:
    """Register the public key if the account does not already carry it.
    Returns the key id to pass as `sshKeyId`, or None if we have no key."""
    if not pubkey:
        return None
    for key in api.ssh_keys():
        existing = str(key.get("publicKey") or key.get("public_key") or "").strip()
        if existing and existing.split()[:2] == pubkey.split()[:2]:
            return str(key.get("id") or key.get("key_id") or "") or None
    created = api.create_ssh_key(name, pubkey)
    return str(created.get("id") or (created.get("data") or {}).get("id") or "") or None


def _wait_for_ssh(api: PrimeAPI, pod_id: str, timeout_s: int, poll_s: int = 15) -> dict:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        pod = api.pod(pod_id) or {}
        status = str(pod.get("status") or "")
        if status != last:
            print(f"  status: {status or '?'}", flush=True)
            last = status
        if status in DEAD_STATUSES:
            raise RuntimeError(f"pod {pod_id} reached {status} before becoming reachable")
        if status == "ACTIVE" and ssh_target(pod):
            return pod
        time.sleep(poll_s)
    raise TimeoutError(f"pod {pod_id} was not reachable within {timeout_s}s")


def cmd_up(api: PrimeAPI, args: argparse.Namespace) -> int:
    sha, clean = git_image_tag()
    image_ref = args.image or f"{IMAGE_REPO}:{sha}"
    if not clean and not args.image:
        print(f"WARNING: working tree is dirty; {image_ref} is the *committed* tree. "
              "Results' provenance is the image, not what is on disk here.")

    # Image parity. PI's `image` is an enum of THEIR environments; ours is only
    # reachable as a console-registered custom template. Booting their ubuntu
    # would run no distrain code at all, so refuse rather than produce a pod that
    # looks fine and cannot be reported.
    if not args.template_id and not args.allow_stock_image:
        print(f"refusing to provision: no --template-id, so the pod would boot Prime "
              f"Intellect's stock '{args.stock_image}' image instead of {image_ref}.\n"
              "  --template-id takes the *image reference* in Prime's own registry, not a\n"
              "  console template id -- it is passed through as `customTemplateId`:\n"
              "      scripts/container.sh push          # ghcr, Docker media types (decisions §20)\n"
              "      prime images push distrain:<sha> --context . --dockerfile Dockerfile\n"
              "      prime images list                 # -> prime/team-<id>/distrain:<sha>\n"
              "  Their registry cannot copy from ghcr.io (`sourceImage` is public-only, so a\n"
              "  private source 401s), so this rebuilds from our Dockerfile. `uv sync\n"
              "  --frozen` against uv.lock keeps the dependency set identical, but the digest\n"
              "  differs from the aurora build -- record both in the session log.\n"
              "  --allow-stock-image boots their Ubuntu instead -- which is the better\n"
              "  route here, not a fallback: a PI pod is a KVM VM with root (decisions\n"
              "  §20), so install Docker and `docker pull` the *aurora* image. That is\n"
              "  byte-identical, where their registry rebuilds. Measure nothing on the\n"
              "  bare stock image itself.")
        return 2

    wallet = api.wallet()
    balance = float(wallet.get("balance_usd") or 0.0)

    existing = find_pod(api.pods(), args.name)
    if existing:
        pod_id = str(existing.get("id"))
        print(f"pod: reusing {pod_id} ({args.name}), status {existing.get('status')}, "
              f"${float(existing.get('priceHr') or 0):.2f}/h")
        target = ssh_target(existing) or ssh_target(_wait_for_ssh(api, pod_id, args.wait_seconds))
        if target:
            print(f"\n  {ssh_command(*target)}\n")
        return 0

    availability = api.availability(gpu_type=args.gpu_type, gpu_count=args.gpu_count)
    offers = matching_offers(availability, args.gpu_type, args.gpu_count,
                             socket=args.socket, provider=args.provider)
    if not offers:
        print(f"no in-stock {args.gpu_count}x {args.gpu_type} "
              f"{args.socket} offer right now.")
        return 1
    offer = offers[0]
    price = offer_price(offer) or 0.0

    refusal = budget_verdict(balance, price, args.max_hours)
    if refusal:
        if args.dry_run:            # a dry run reports the problem, it isn't blocked by it
            print(f"NOTE: {refusal}")
        elif not args.force:
            sys.exit(f"refusing to provision: {refusal}  (--force overrides)")
        else:
            print(f"WARNING (--force): {refusal}")

    pubkey = read_pubkey(args.ssh_key)
    if not pubkey:
        print("WARNING: no SSH public key found; the pod would boot unreachable. "
              "Pass --ssh-key.")

    started = datetime.now(UTC)
    deadline = deadline_iso(started, args.max_hours)

    plan = {
        "name": args.name,
        "image": image_ref if args.template_id else f"(stock {args.stock_image})",
        "template_id": args.template_id,
        "gpu": f"{args.gpu_count}x {args.gpu_type} {offer.get('socket')}",
        "provider": offer.get("provider"),
        "data_center": offer.get("dataCenter"),
        "country": offer.get("country"),
        "cloud_id": offer.get("cloudId"),
        "disk_gb": args.disk_gb,
        "price_per_hr": price,
        "max_hours": args.max_hours,
        "deadline_utc": deadline,
        "estimated_ceiling_usd": round(price * args.max_hours, 2),
        "balance_usd": balance,
    }
    print("\nplan:")
    for key, value in plan.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        print("\n--dry-run: nothing rented.")
        return 0

    pod_config: dict = {
        "name": args.name,
        "cloudId": offer.get("cloudId"),
        "gpuType": args.gpu_type,
        "socket": offer.get("socket"),
        "gpuCount": args.gpu_count,
        "diskSize": args.disk_gb,
        "maxPrice": round(price * 1.05, 4),   # the offer's price, not an open cheque
        "autoRestart": False,                 # a restart after our deadline would bill on
    }
    if args.template_id:
        pod_config["image"] = "custom_template"
        pod_config["customTemplateId"] = args.template_id
        # Some pod types reject envVars outright ("Environment variables are not
        # allowed for this request"). The deadline is informational -- `guard`
        # enforces the ceiling -- so it must never be the reason a pod fails to
        # create. Only send it where it is accepted.
        pod_config["envVars"] = [{"key": "DISTRAIN_DEADLINE_UTC", "value": deadline}]
    else:
        pod_config["image"] = args.stock_image
    if offer.get("dataCenter"):
        pod_config["dataCenterId"] = offer["dataCenter"]
    if offer.get("country"):
        pod_config["country"] = offer["country"]

    key_id = _ensure_ssh_key(api, pubkey, args.ssh_key_name)
    if key_id:
        pod_config["sshKeyId"] = key_id

    payload = {"pod": pod_config, "provider": {"type": offer.get("provider")}}
    created = api.create_pod(payload)
    pod_id = str(created.get("id") or (created.get("data") or {}).get("id") or "")
    if not pod_id:
        sys.exit(f"pod creation returned no id: {created}")
    print(f"\npod: created {pod_id} -- billing from now, ${price:.2f}/h")

    try:                                      # teardown on exception: §9's kill-switch
        ready = _wait_for_ssh(api, pod_id, args.wait_seconds)
    except BaseException as exc:              # includes KeyboardInterrupt
        if args.keep_on_error:
            print(f"\n{type(exc).__name__}: {exc}\npod {pod_id} LEFT RUNNING "
                  f"(--keep-on-error) -- it is billing at ${price:.2f}/h.")
            raise
        print(f"\n{type(exc).__name__}: {exc}\nterminating {pod_id} so it cannot bill quietly.")
        try:
            api.terminate_pod(pod_id)
        except Exception as cleanup_exc:       # noqa: BLE001 -- never mask the original
            print(f"WARNING: teardown itself failed ({cleanup_exc}). "
                  f"TERMINATE {pod_id} BY HAND -- it is billing.")
        raise

    user, ip, port = ssh_target(ready)
    session = {**plan, "pod_id": pod_id,
               "started_utc": started.replace(microsecond=0).isoformat(),
               "ssh": ssh_command(user, ip, port), "user": user, "ip": ip, "port": port}
    write_session(session, args.session_file)

    print(f"\n  {ssh_command(user, ip, port)}\n")
    print(f"ceiling {args.max_hours:g} h -> terminate by {deadline} UTC "
          f"(~${price * args.max_hours:.2f}). Start the kill-switch now:")
    print(f"  uv run --script {Path(__file__).relative_to(REPO_ROOT)} guard &")
    return 0


def cmd_guard(api: PrimeAPI, args: argparse.Namespace) -> int:
    """The cost kill-switch (docs/decisions.md §9): a hard wall-clock ceiling that
    terminates the pod, plus a balance watch, so an unattended run cannot outlive
    either. Terminating loses the disk -- pull artifacts before the deadline."""
    session = read_session(args.session_file)
    pod_id = args.pod_id or session.get("pod_id")
    if not pod_id:
        sys.exit(f"no pod id: pass --pod-id, or run `up` first ({args.session_file} is empty)")

    deadline = args.deadline or session.get("deadline_utc")
    if args.max_hours:
        deadline = deadline_iso(datetime.now(UTC), args.max_hours)
    if not deadline:
        sys.exit("no deadline: pass --max-hours or --deadline")
    deadline_dt = datetime.fromisoformat(deadline)
    if deadline_dt.tzinfo is None:
        deadline_dt = deadline_dt.replace(tzinfo=UTC)

    print(f"guarding {pod_id}: terminate at {deadline_dt.isoformat()} UTC, "
          f"balance floor ${args.min_balance:.2f}, poll {args.poll_seconds}s")
    while True:
        now = datetime.now(UTC)
        pod = api.pod(pod_id)
        if not pod or str(pod.get("status")) in DEAD_STATUSES:
            print(f"[{now:%H:%M:%S}] pod {pod_id} is gone; nothing to guard.")
            return 0

        cost = float(pod.get("priceHr") or 0)
        balance = float(api.wallet().get("balance_usd") or 0)
        left = (deadline_dt - now).total_seconds() / 3600
        print(f"[{now:%H:%M:%S}] ${balance:.2f} left, ${cost:.2f}/h, runway "
              f"{runway_hours(balance, cost):.1f} h, ceiling in {left:.2f} h", flush=True)

        if now >= deadline_dt:
            print(f"ceiling reached -- terminating {pod_id}.")
            api.terminate_pod(pod_id)
            return 0
        if balance <= args.min_balance:
            print(f"balance ${balance:.2f} at or below the ${args.min_balance:.2f} floor "
                  f"-- terminating {pod_id} before the credit runs out mid-run.")
            api.terminate_pod(pod_id)
            return 0
        time.sleep(args.poll_seconds)


def cmd_ssh(api: PrimeAPI, args: argparse.Namespace) -> int:
    session = read_session(args.session_file)
    pod_id = args.pod_id or session.get("pod_id")
    if not pod_id:
        sys.exit("no session; run `up` first")
    pod = api.pod(pod_id)
    if not pod:
        sys.exit(f"pod {pod_id} not found")
    target = ssh_target(pod)
    if not target:
        sys.exit(f"pod {pod_id} is {pod.get('status')} and has no ssh connection yet")
    line = ssh_command(*target)
    if args.exec:
        return subprocess.call(line.split() + (args.command or []))
    print(line)
    return 0


def cmd_down(api: PrimeAPI, args: argparse.Namespace) -> int:
    session = read_session(args.session_file)
    pod_id = args.pod_id or session.get("pod_id")
    if not pod_id:
        existing = find_pod(api.pods(), args.name)
        pod_id = str(existing.get("id")) if existing else None
    if not pod_id:
        print("no pod to terminate.")
        return 0
    pod = api.pod(pod_id)
    if not pod or str(pod.get("status")) in DEAD_STATUSES:
        print(f"pod {pod_id} is already {pod.get('status') if pod else 'gone'}.")
        return 0
    api.terminate_pod(pod_id)
    print(f"terminated {pod_id}.")
    if args.session_file.exists():
        args.session_file.unlink()
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu-type", default=GPU_TYPE,
                        help=f"PI GPU enum name (default {GPU_TYPE})")
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT)
    parser.add_argument("--socket", default=SOCKET,
                        help=f"GPU socket, or 'any' to stop pinning it (default {SOCKET}; "
                             "PCIe is not comparable to the SXM4 anchor)")
    parser.add_argument("--provider", default=None,
                        help="pin an upstream provider (lambdalabs, vultr, ...)")
    parser.add_argument("--session-file", type=Path, default=SESSION_FILE)
    parser.add_argument("--name", default="distrain")
    parser.add_argument("--team-id", default=None,
                        help="bill this team's wallet (default: PRIME_TEAM_ID, .env, "
                             "or the `prime` CLI's config; --team-id '' forces personal)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="balance, pods, registry credentials")

    verify = subparsers.add_parser("verify", help="is anything still billing?")
    verify.add_argument("--strict", action="store_true",
                        help="exit non-zero if anything is live")

    subparsers.add_parser("avail", help="live stock for the GPU shape")

    up = subparsers.add_parser("up", help="pick an offer, create the pod, wait for SSH")
    up.add_argument("--max-hours", type=float, default=4.0)
    up.add_argument("--disk-gb", type=int, default=DISK_GB)
    up.add_argument("--image", default=None,
                    help="image reference recorded in the plan (default: GHCR at HEAD)")
    up.add_argument("--template-id", default=None,
                    help="PI custom template id -- note templates have no API and could "
                         "not be created for this account (decisions §20)")
    up.add_argument("--allow-stock-image", action="store_true",
                    help="boot PI's Ubuntu, then run our container inside it (the pod is "
                         "a root KVM VM); measure nothing on the bare image")
    up.add_argument("--stock-image", default="ubuntu_22_cuda_12",
                    help="which stock image, when --allow-stock-image is given")
    up.add_argument("--ssh-key", type=Path, default=None)
    up.add_argument("--ssh-key-name", default="distrain-aurora")
    up.add_argument("--wait-seconds", type=int, default=900)
    up.add_argument("--dry-run", action="store_true", help="print the plan, rent nothing")
    up.add_argument("--force", action="store_true", help="provision despite the budget refusal")
    up.add_argument("--keep-on-error", action="store_true",
                    help="do NOT terminate if provisioning fails (it keeps billing)")

    guard = subparsers.add_parser("guard", help="wall-clock ceiling + balance watch")
    guard.add_argument("--pod-id", default=None)
    guard.add_argument("--deadline", default=None)
    guard.add_argument("--max-hours", type=float, default=None,
                       help="reset the deadline to now + this many hours")
    guard.add_argument("--min-balance", type=float, default=2.0)
    guard.add_argument("--poll-seconds", type=int, default=60)

    ssh = subparsers.add_parser("ssh", help="print (or run) the ssh command")
    ssh.add_argument("--pod-id", default=None)
    ssh.add_argument("--exec", action="store_true", help="run it instead of printing it")
    ssh.add_argument("command", nargs="*", help="command to run over ssh with --exec")

    down = subparsers.add_parser("down", help="terminate the pod")
    down.add_argument("--pod-id", default=None)

    return parser


COMMANDS = {
    "status": cmd_status, "verify": cmd_verify, "avail": cmd_avail, "up": cmd_up,
    "guard": cmd_guard, "ssh": cmd_ssh, "down": cmd_down,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `--team-id ''` is the explicit way back to the personal wallet; absent, the
    # discovered team wins, because that is where a console top-up actually goes.
    team = args.team_id if args.team_id is not None else load_team_id()
    api = PrimeAPI(load_api_key(), team_id=team or None)
    return COMMANDS[args.command](api, args)


if __name__ == "__main__":
    sys.exit(main())
