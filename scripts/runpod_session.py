#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["runpod==1.12.0"]
# ///
"""Provision / inspect / tear down the rented box a session runs on (RunPod).

Everything the 8-GPU runbook does by hand in the console, made idempotent and
repeatable: ensure a network volume exists, ensure a pod of the wanted GPU type
exists, boot it from the pinned GHCR image with `scripts/pod-entry.sh`, wait for
SSH, and hand back the ssh line. Re-running `up` on a live session is a no-op
that re-prints the ssh line -- it never rents a second box.

    uv run --script scripts/runpod_session.py status
    uv run --script scripts/runpod_session.py avail            # per-DC 8xA100 stock
    uv run --script scripts/runpod_session.py up --dry-run     # the plan, free
    uv run --script scripts/runpod_session.py up --max-hours 8
    uv run --script scripts/runpod_session.py guard            # the cost kill-switch
    uv run --script scripts/runpod_session.py ssh [--exec]
    uv run --script scripts/runpod_session.py down
    uv run --script scripts/runpod_session.py verify [--strict]   # is anything billing?

It runs on the `runpod` SDK (1.12.x, GraphQL under the hood). The SDK has no
wrapper for network volumes, templates, registry credentials or the account
balance, so those four go through the SDK's own `run_graphql_query`.

Deliberate, load-bearing choices (docs/decisions.md §9, the 2026-08-16 session):

- **The pod's own volume is always 0 GB.** The API defaults it to 20 GB mounted
  at /workspace, which would shadow the baked code and silently run whatever was
  on the volume -- faking the image-parity check rather than failing it.
- **A network volume never mounts at /workspace**, same reason; /data by default,
  and any attempt to mount at /workspace is refused.
- **Teardown on exception.** Anything that fails after the pod exists terminates
  it before exiting, so a half-provisioned box cannot bill quietly.
- **`up` refuses to start what the balance cannot finish** -- credit exhaustion
  terminated a nearly-converged run once (session log §6a); `guard` then enforces
  the wall-clock ceiling and watches the balance for the rest of the session.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import runpod
from runpod.api.graphql import run_graphql_query

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_FILE = REPO_ROOT / "out" / "runpod" / "session.json"

GPU_TYPE = "NVIDIA A100-SXM4-80GB"     # the runbook's box; A100 PCIe / H100 are fallbacks
GPU_COUNT = 8
CONTAINER_DISK_GB = 80
POD_PORTS = "22/tcp"                   # SSH over exposed TCP; pod-entry.sh starts sshd
START_CMD = "/workspace/scripts/pod-entry.sh"   # --start-cmd '' boots a stock image's own CMD
REGISTRY_AUTH_NAME = "GitHub packages"  # RunPod's own credential for private GHCR pulls
VOLUME_MOUNT_PATH = "/data"
IMAGE_REPO = "ghcr.io/adamdivak/distrain"
VOLUME_USD_PER_GB_MONTH = 0.07         # list price under 1 TB; only used to print an estimate

# Fallback only -- the live list comes from `RunpodAPI.data_centers()`. This
# snapshot was 28 entries against the API's 49, so half the fleet was never
# scanned and `avail` could report "no capacity" for a GPU that was rentable.
# Kept solely so a failed query degrades instead of provisioning nothing.
FALLBACK_DATA_CENTERS = [
    "EU-RO-1", "CA-MTL-1", "EU-SE-1", "US-IL-1", "EUR-IS-1", "EU-CZ-1", "US-TX-3",
    "EUR-IS-2", "US-KS-2", "US-GA-2", "US-WA-1", "US-TX-1", "CA-MTL-3", "EU-NL-1",
    "US-TX-4", "US-CA-2", "US-NC-1", "OC-AU-1", "US-DE-1", "EUR-IS-3", "CA-MTL-2",
    "AP-JP-1", "EUR-NO-1", "EU-FR-1", "US-KS-3", "US-GA-1", "AP-IN-1", "US-MD-1",
]


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #

def load_api_key(env_file: Path = REPO_ROOT / ".env") -> str:
    """RUNPOD_API_KEY from the environment, else from the gitignored .env."""
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "RUNPOD_API_KEY":
                key = value.strip().strip("'\"")
                break
    if not key:
        sys.exit(f"no RUNPOD_API_KEY in the environment or {env_file}")
    return key


# --------------------------------------------------------------------------- #
# API surface -- one seam, so the tests can drive the whole script with a fake
# --------------------------------------------------------------------------- #

class RunpodAPI:
    """Thin wrapper over the SDK. Calls the SDK exposes go through it; the four
    it doesn't cover (volumes, templates, registry creds, balance) go through the
    SDK's own GraphQL transport."""

    def gql(self, query: str) -> dict:
        return run_graphql_query(query)["data"]

    # -- account ----------------------------------------------------------- #

    def account(self) -> dict:
        return self.gql(
            "query { myself { id pubKey clientBalance currentSpendPerHr "
            "networkVolumes { id name size dataCenterId } "
            "containerRegistryCreds { id name } "
            "podTemplates { id name imageName } } }"
        )["myself"]

    # -- pods -------------------------------------------------------------- #

    def pods(self) -> list[dict]:
        return runpod.get_pods()

    def pod(self, pod_id: str) -> dict | None:
        return runpod.get_pod(pod_id)

    def pod_runtime(self, pod_id: str) -> dict:
        """`podHostId` and container uptime -- neither is in the SDK's own query.

        `podHostId` is the username of RunPod's SSH proxy
        (`<podHostId>@ssh.runpod.io`), which execs into any *running* container.
        That is the only way onto a host that exposes no public port, which is
        every community host this project has drawn (2026-08-22).
        """
        data = self.gql(
            'query { pod(input: {podId: "%s"}) { id desiredStatus '
            "runtime { uptimeInSeconds } machine { podHostId } } }" % pod_id
        )
        return data.get("pod") or {}

    def create_pod(self, **kwargs) -> dict:
        return runpod.create_pod(**kwargs)

    def resume_pod(self, pod_id: str, gpu_count: int) -> dict:
        return runpod.resume_pod(pod_id, gpu_count)

    def terminate_pod(self, pod_id: str) -> None:
        runpod.terminate_pod(pod_id)

    def endpoints(self) -> list[dict]:
        """Serverless endpoints. This project creates none -- one existing would
        be an accident, and accidents bill."""
        try:
            return runpod.get_endpoints() or []
        except runpod.error.QueryError as exc:   # an account without serverless enabled
            print(f"warning: could not list serverless endpoints ({exc})")
            return []

    # -- templates and registry credentials -------------------------------- #

    def create_template(self, **kwargs) -> dict:
        return runpod.create_template(**kwargs)

    def create_registry_auth(self, name: str, username: str, password: str) -> dict:
        return runpod.create_container_registry_auth(name, username, password)

    # -- network volumes (no SDK wrapper) ---------------------------------- #

    def create_volume(self, name: str, size_gb: int, data_center_id: str) -> dict:
        return self.gql(
            f'mutation {{ createNetworkVolume(input: {{name: "{name}", size: {size_gb}, '
            f'dataCenterId: "{data_center_id}"}}) {{ id name size dataCenterId }} }}'
        )["createNetworkVolume"]

    def delete_volume(self, volume_id: str) -> None:
        self.gql(f'mutation {{ deleteNetworkVolume(input: {{id: "{volume_id}"}}) }}')

    # -- availability ------------------------------------------------------ #

    def data_centers(self) -> list[str]:
        """Every data center id the API currently reports, sorted.

        Queried rather than hardcoded: a stale snapshot silently shrinks the
        search, and a data center that is missing from the list is
        indistinguishable from one with no capacity. Falls back to the frozen
        list only if the query fails, so a network blip degrades the search
        instead of emptying it.
        """
        try:
            dcs = self.gql("query { dataCenters { id } }")["dataCenters"]
            ids = sorted(d["id"] for d in dcs if d.get("id"))
            return ids or list(FALLBACK_DATA_CENTERS)
        except Exception:  # noqa: BLE001 -- availability must not hard-fail here
            return list(FALLBACK_DATA_CENTERS)

    def gpu_price(self, gpu_type: str, gpu_count: int, data_center_id: str | None = None,
                  secure: bool | None = None) -> dict:
        """Two different numbers, and mixing them up costs money.

        `lowestPrice` is the *availability* signal -- it is null in a data center
        with no capacity -- but it is the lowest across cloud types, i.e. the
        community rate. This script deploys `cloudType: SECURE`, which bills
        `securePrice` per GPU: measured, a 3090 pod billed $0.50/h against a
        $0.22 `lowestPrice`, and 8x A100 bills 8 x $1.59 = $12.72/h, exactly what
        the 2026-08-16 session paid. So availability comes from one field and the
        money from the other.

        `data_center_id` is not merely a narrowing filter: **community hosts
        carry no data center id**, so every per-data-center query returns null
        for them no matter how many data centers are swept. Asking only per-DC
        therefore reports "no capacity" for GPUs the console rents happily --
        measured 2026-08-18, when 8x 4090 was null in all 49 data centers while
        the unscoped query returned $2.72/h. Pass `secure=False` and no data
        center to see community stock."""
        dc = f', dataCenterId: "{data_center_id}"' if data_center_id else ""
        sec = "" if secure is None else f", secureCloud: {'true' if secure else 'false'}"
        types = self.gql(
            f'query {{ gpuTypes(input: {{id: "{gpu_type}"}}) {{ id maxGpuCount '
            f"securePrice communityPrice "
            f"lowestPrice(input: {{gpuCount: {gpu_count}{dc}{sec}}}) "
            "{ uninterruptablePrice minimumBidPrice stockStatus } } }"
        )["gpuTypes"]
        if not types:
            sys.exit(f"unknown GPU type: {gpu_type}")
        return types[0]


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested offline -- a bug found here costs a minute, found on
# the rented box it costs the hourly rate)
# --------------------------------------------------------------------------- #

def find_pod(pods: list[dict], name: str) -> dict | None:
    """The pod this session owns, by exact name. Terminated pods are not returned
    by the API at all, so anything found is either RUNNING or stopped (EXITED)."""
    return next((p for p in pods if p.get("name") == name), None)


def find_volume(volumes: list[dict], name: str, min_size_gb: int) -> dict | None:
    return next(
        (v for v in volumes if v.get("name") == name and v.get("size", 0) >= min_size_gb), None
    )


def find_by_name(items: list[dict], name: str) -> dict | None:
    return next((i for i in items if i.get("name") == name), None)


def ssh_target(pod: dict) -> tuple[str, int] | None:
    """(ip, public_port) for the pod's exposed port 22, or None until it maps."""
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port["ip"], int(port["publicPort"])
    return None


def ssh_command(ip: str, port: int) -> str:
    return f"ssh -p {port} root@{ip}"


def runway_hours(balance: float, cost_per_hr: float) -> float:
    """Hours of spend the remaining credit covers. inf when nothing is running."""
    return float("inf") if cost_per_hr <= 0 else balance / cost_per_hr


def budget_verdict(balance: float, price_per_hr: float, max_hours: float) -> str | None:
    """Refusal message when the balance cannot fund the whole ceiling, else None.
    RunPod *terminates* pods on a drained balance -- the container disk dies with
    them -- so starting a run the credit cannot finish is a way to pay for
    nothing (session log §6a)."""
    need = price_per_hr * max_hours
    if balance < need:
        return (
            f"balance ${balance:.2f} cannot fund {max_hours:g} h at ${price_per_hr:.2f}/h "
            f"(${need:.2f}). Top up, or lower --max-hours to "
            f"{runway_hours(balance, price_per_hr):.1f} h. RunPod terminates pods when "
            "credit runs out and the container disk goes with them."
        )
    return None


def deadline_iso(started: datetime, max_hours: float) -> str:
    return (started + timedelta(hours=max_hours)).replace(microsecond=0).isoformat()


def read_pubkey(path: Path | None) -> str:
    """The public key pod-entry.sh installs. RunPod injects the account key as
    $PUBLIC_KEY for console-created pods; over the API we pass it ourselves."""
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
    a session produces, so a dirty tree is worth a warning.

    Returns ("unknown", False) where there is no git repo -- notably *inside* the
    image, which bakes the code but not `.git`. Raising there made the parity
    suite fail on the pod for a reason that has nothing to do with the code
    being checked.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False
    return sha, not dirty


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

def cmd_status(api: RunpodAPI, args: argparse.Namespace) -> int:
    account = api.account()
    pods = api.pods()
    balance = float(account.get("clientBalance") or 0.0)
    spend = sum(float(p.get("costPerHr") or 0) for p in pods
                if p.get("desiredStatus") == "RUNNING")

    print(f"balance ${balance:.2f} | running spend ${spend:.2f}/h", end="")
    print(f" | runway {runway_hours(balance, spend):.1f} h" if spend else " | nothing running")

    print(f"\npods ({len(pods)}):")
    for pod in pods:
        gpu = (pod.get("machine") or {}).get("gpuDisplayName", "?")
        target = ssh_target(pod)
        print(f"  {pod['id']}  {pod.get('name')}  {pod.get('desiredStatus')}  "
              f"{pod.get('gpuCount')}x {gpu}  ${float(pod.get('costPerHr') or 0):.2f}/h  "
              f"up {int(pod.get('uptimeSeconds') or 0) // 60} min")
        print(f"      image {pod.get('imageName')}")
        if target:
            print(f"      {ssh_command(*target)}")
    if not pods:
        print("  (none)")

    volumes = account.get("networkVolumes") or []
    print(f"\nnetwork volumes ({len(volumes)}):")
    for vol in volumes:
        print(f"  {vol['id']}  {vol['name']}  {vol['size']} GB  {vol['dataCenterId']}")
    if not volumes:
        print("  (none)")

    creds = account.get("containerRegistryCreds") or []
    print("\nregistry credentials: " + (", ".join(c["name"] for c in creds) or "(none)"))

    session = read_session(args.session_file)
    if session.get("deadline_utc"):
        print(f"\nsession {session.get('pod_id')} deadline {session['deadline_utc']} UTC")
    return 0


def cmd_verify(api: RunpodAPI, args: argparse.Namespace) -> int:
    """The end-of-session check: is anything still costing money?

    Two tiers, because they are different kinds of mistake.

    *Metered by the hour* -- running pods, stopped pods (no GPU charge, but their
    disks still bill), serverless endpoints -- is a leak: nothing here is ever
    meant to outlive a session. Any of it fails the check.

    *Metered by the month* -- network volumes, ~$0.07/GB -- is usually deliberate:
    a volume is the thing that survives a pod on purpose, and it is three orders
    of magnitude cheaper than an idle 8×A100. So it is reported, loudly and with
    a price, but it does not fail the check unless `--strict` says the account
    should be empty (the end of the project, not the end of a session).
    """
    strict = getattr(args, "strict", False)
    account = api.account()
    pods = api.pods()
    running = [p for p in pods if p.get("desiredStatus") == "RUNNING"]
    stopped = [p for p in pods if p.get("desiredStatus") not in ("RUNNING", "TERMINATED")]
    endpoints = api.endpoints()
    volumes = account.get("networkVolumes") or []
    hourly = sum(float(p.get("costPerHr") or 0) for p in running)
    reported_spend = float(account.get("currentSpendPerHr") or 0)

    def describe(pod: dict) -> str:
        gpu = (pod.get("machine") or {}).get("gpuDisplayName", "?")
        return (f"{pod['id']}  {pod.get('name')}  {pod.get('gpuCount')}x {gpu}  "
                f"${float(pod.get('costPerHr') or 0):.2f}/h")

    print("metered by the hour -- must be empty after a session:")
    print(f"  running pods:         {len(running)}")
    for pod in running:
        print(f"    {describe(pod)}")
    print(f"  stopped pods:         {len(stopped)}    (no GPU charge; disks still bill)")
    for pod in stopped:
        print(f"    {describe(pod)}")
    print(f"  serverless endpoints: {len(endpoints)}")
    for endpoint in endpoints:
        print(f"    {endpoint.get('id')}  {endpoint.get('name')}")
    print(f"  account currentSpendPerHr: ${reported_spend:.2f}/h")

    monthly = sum(float(v.get("size") or 0) * VOLUME_USD_PER_GB_MONTH for v in volumes)
    print(f"\nmetered by the month -- persists on purpose ({len(volumes)}):")
    for vol in volumes:
        cost = float(vol.get("size") or 0) * VOLUME_USD_PER_GB_MONTH
        print(f"  {vol['id']}  {vol['name']}  {vol['size']} GB  {vol['dataCenterId']}  "
              f"~${cost:.2f}/mo")
    if not volumes:
        print("  (none)")

    print(f"\nbalance ${float(account.get('clientBalance') or 0):.2f}")

    leaks = []
    if running:
        leaks.append(f"{len(running)} running pod(s) at ${hourly:.2f}/h")
    if stopped:
        leaks.append(f"{len(stopped)} stopped pod(s) still billing disk")
    if endpoints:
        leaks.append(f"{len(endpoints)} serverless endpoint(s)")
    if reported_spend > 0 and not running:
        # The account's own number disagrees with the pod list. Measured: after a
        # termination it keeps reporting the old rate for ~75 s, so re-read it
        # once before calling it a leak -- otherwise every `down` fails its own
        # check. Still a failure if it persists: trust the money over the list.
        settle = getattr(args, "settle_seconds", 45)
        if settle:
            print(f"\naccount reports ${reported_spend:.2f}/h with no running pod listed; "
                  f"re-reading in {settle}s (billing lags a termination by ~a minute)...")
            time.sleep(settle)
            reported_spend = float(api.account().get("currentSpendPerHr") or 0)
            print(f"  now ${reported_spend:.2f}/h")
        if reported_spend > 0:
            leaks.append(f"account reports ${reported_spend:.2f}/h with no running pod listed")
    if strict and volumes:
        leaks.append(f"{len(volumes)} network volume(s) at ~${monthly:.2f}/mo (--strict)")

    if leaks:
        print("\nSTILL BILLING: " + "; ".join(leaks))
        print("Terminate with `down` (add --delete-volume for the volume), then re-run this.")
        return 1

    if volumes:
        print(f"\nCLEAN by the hour: nothing metered hourly is alive. {len(volumes)} network "
              f"volume(s) remain at ~${monthly:.2f}/mo -- kept on purpose; `down --delete-volume` "
              "or `verify --strict` if they should be gone too.")
    else:
        print("\nCLEAN: nothing on this account is costing money.")
    return 0


def cmd_avail(api: RunpodAPI, args: argparse.Namespace) -> int:
    """Which data centers can actually rent `--gpu-count` of the GPU type right
    now. This is what picks the data center -- and, with a network volume, the
    volume must live in the same one, so the choice is sticky."""
    overall = api.gpu_price(args.gpu_type, args.gpu_count)
    print(f"{args.gpu_type} x{args.gpu_count}  max/host {overall.get('maxGpuCount')}  "
          f"secure ${secure_price(overall, args.gpu_count):.2f}/h  "
          f"(community ${float(overall.get('communityPrice') or 0) * args.gpu_count:.2f}/h)\n")

    found = []
    for dc in (args.data_centers or api.data_centers()):
        low = api.gpu_price(args.gpu_type, args.gpu_count, dc, secure=True).get("lowestPrice") or {}
        if low.get("uninterruptablePrice"):
            found.append(dc)
            print(f"  SECURE     {dc:10s} capacity  {low.get('stockStatus') or ''}")
    if not found:
        print("  SECURE     no data center reports capacity for that GPU count.")

    # Community hosts have no data center id, so the loop above can never see
    # them; ask unscoped or they look permanently sold out (see gpu_price).
    comm = api.gpu_price(args.gpu_type, args.gpu_count, secure=False).get("lowestPrice") or {}
    if comm.get("uninterruptablePrice"):
        print(f"  COMMUNITY  (no data center)  capacity  {comm.get('stockStatus') or ''}  "
              f"-- provision with --cloud-type COMMUNITY --volume-gb 0")
        found.append("COMMUNITY")
    else:
        print("  COMMUNITY  no capacity for that GPU count.")
    return 0 if found else 1


def secure_price(gpu_type_info: dict, gpu_count: int) -> float:
    """What a SECURE pod actually bills per hour: per-GPU secure price x count."""
    return float(gpu_type_info.get("securePrice") or 0) * gpu_count


def community_price(gpu_type_info: dict, gpu_count: int) -> float:
    """What a COMMUNITY pod bills per hour: per-GPU community price x count."""
    return float(gpu_type_info.get("communityPrice") or 0) * gpu_count


def _pick_placement(api: RunpodAPI, args: argparse.Namespace) -> tuple[str | None, float]:
    """(data center or None, the hourly price that cloud type will bill).

    SECURE pods are placed in a named data center, so the per-DC query is both
    the availability signal and the placement decision. COMMUNITY hosts carry
    no data center id -- every per-DC query is null for them -- so there is
    nothing to pick and RunPod places the pod itself; `None` says exactly that.
    Asking per-DC for a community pod is what made 8x 4090 look sold out while
    the console rented it (see `RunpodAPI.gpu_price`).
    """
    if args.skip_capacity_check:
        # The precheck is advisory, and measured 2026-08-18 it is not even
        # reliable: 8x 4090 was null in all 49 data centers and under both cloud
        # filters while the console rented one at the secure price. RunPod's
        # deploy call does its own placement, so let it be the capacity signal --
        # a failed deploy costs nothing, an unrentable precheck costs the session.
        info = api.gpu_price(args.gpu_type, args.gpu_count)
        price = (community_price if args.cloud_type == "COMMUNITY" else secure_price)(
            info, args.gpu_count)
        print(f"capacity precheck skipped; letting the deploy call place the pod "
              f"({args.cloud_type}, ${price:.2f}/h)")
        return args.data_center, price

    if args.cloud_type == "COMMUNITY":
        if args.data_center:
            sys.exit("--data-center cannot be combined with --cloud-type COMMUNITY: "
                     "community hosts carry no data center id.")
        info = api.gpu_price(args.gpu_type, args.gpu_count, secure=False)
        if (info.get("lowestPrice") or {}).get("uninterruptablePrice"):
            return None, community_price(info, args.gpu_count)
        sys.exit(f"no capacity for {args.gpu_count}x {args.gpu_type} on COMMUNITY hosts "
                 f"right now (check `avail`)")

    candidates = ([args.data_center] if args.data_center
                  else (args.data_centers or api.data_centers()))
    for dc in candidates:
        info = api.gpu_price(args.gpu_type, args.gpu_count, dc, secure=True)
        if (info.get("lowestPrice") or {}).get("uninterruptablePrice"):
            return dc, secure_price(info, args.gpu_count)
    sys.exit(f"no capacity for {args.gpu_count}x {args.gpu_type} on SECURE hosts in "
             f"{'/'.join(candidates) if args.data_center else 'any data center'} right now "
             f"(check `avail`; --cloud-type COMMUNITY searches the hosts that have no "
             f"data center id, and --skip-capacity-check lets the deploy call decide -- "
             f"the query has known false negatives)")


def _ensure_volume(api: RunpodAPI, args: argparse.Namespace, account: dict,
                   data_center: str) -> dict | None:
    """The network volume, created if missing. Returns None when volumes are off.

    A volume pins the pod to its data center, so it is created only after the
    data center is chosen -- and reused only if it is in that same one."""
    if args.volume_gb <= 0:
        return None
    if args.volume_mount_path == "/workspace":
        sys.exit("refusing to mount a network volume at /workspace: it shadows the "
                 "baked code and fakes the image-parity check. Use /data.")

    existing = find_volume(account.get("networkVolumes") or [], args.volume_name, args.volume_gb)
    if existing:
        if existing["dataCenterId"] != data_center:
            sys.exit(f"volume {existing['name']} is in {existing['dataCenterId']} but the pod "
                     f"would land in {data_center}. Pin with --data-center "
                     f"{existing['dataCenterId']}, or drop the volume (--volume-gb 0).")
        print(f"volume: reusing {existing['id']} ({existing['size']} GB, {data_center})")
        return existing

    if args.dry_run:
        print(f"volume: would create {args.volume_name} ({args.volume_gb} GB, {data_center})")
        return {"id": "<dry-run>", "name": args.volume_name, "size": args.volume_gb,
                "dataCenterId": data_center}

    created = api.create_volume(args.volume_name, args.volume_gb, data_center)
    print(f"volume: created {created['id']} ({args.volume_gb} GB, {data_center}) "
          f"-- billed until deleted (`down --delete-volume`)")
    return created


def _ensure_template(api: RunpodAPI, args: argparse.Namespace, account: dict,
                     image: str) -> dict:
    """A template is how a pod gets a private-registry pull credential: the pod
    deploy call in the SDK takes a template id, not a registry auth id."""
    # The suffix keeps a public (no-credential) template from colliding with an
    # authenticated one of the same image tag -- reuse matches on name, and
    # silently inheriting the wrong credential is exactly the failure above.
    name = f"distrain-{image.rsplit(':', 1)[-1]}"
    if not args.registry_auth_name:
        name += "-public"
    existing = next((t for t in account.get("podTemplates") or []
                     if t.get("name") == name and t.get("imageName") == image), None)
    if existing:
        print(f"template: reusing {existing['id']} ({name})")
        return existing

    # A public image must be pulled with *no* credential. Attaching one anyway is
    # not harmless: RunPod hands the named credential to whatever registry the
    # image names, so a GHCR credential on a docker.io image is an
    # IMAGE_AUTH_ERROR ("incorrect username or password") and the pod is stopped
    # before its container ever starts (2026-08-22, three pods lost to this).
    if not args.registry_auth_name:
        if args.dry_run:
            print(f"template: would create {name} -> {image} (public, no credential)")
            return {"id": "<dry-run>", "name": name, "imageName": image}
        template = api.create_template(
            name=name, image_name=image, docker_start_cmd=args.start_cmd,
            container_disk_in_gb=args.container_disk_gb, volume_in_gb=0,
            ports=POD_PORTS, is_serverless=False,
        )
        print(f"template: created {template['id']} ({name}, public image)")
        return template

    auth = find_by_name(account.get("containerRegistryCreds") or [], args.registry_auth_name)
    if not auth and args.ghcr_token:
        auth = api.create_registry_auth(args.registry_auth_name, args.ghcr_user, args.ghcr_token)
        print(f"registry auth: created {auth['id']} ({args.registry_auth_name})")
    if not auth:
        sys.exit(f"no container registry credential named {args.registry_auth_name!r}. The GHCR "
                 "package is private: pass --ghcr-token <PAT with read:packages> once, or add "
                 "the credential in the RunPod console.")

    if args.dry_run:
        print(f"template: would create {name} -> {image} (auth {auth['id']})")
        return {"id": "<dry-run>", "name": name, "imageName": image}

    template = api.create_template(
        name=name,
        image_name=image,
        docker_start_cmd=args.start_cmd,
        container_disk_in_gb=args.container_disk_gb,
        volume_in_gb=0,                      # never a pod volume at /workspace
        ports=POD_PORTS,
        is_serverless=False,
        registry_auth_id=auth["id"],
    )
    print(f"template: created {template['id']} ({name})")
    return template


def _wait_for_ssh(api: RunpodAPI, pod_id: str, timeout_s: int, poll_s: int = 10) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pod = api.pod(pod_id) or {}
        if ssh_target(pod):
            return pod
        time.sleep(poll_s)
    raise TimeoutError(f"pod {pod_id} never exposed port 22 within {timeout_s}s")


def proxy_ssh_command(pod_host_id: str) -> str:
    """RunPod's SSH proxy. `-tt` because the proxy refuses a session without a PTY."""
    return f"ssh -tt {pod_host_id}@ssh.runpod.io"


def _wait_for_container(api: RunpodAPI, pod_id: str, timeout_s: int,
                        poll_s: int = 10) -> str:
    """Block until the container is actually running; return its proxy host id.

    A pod is RUNNING from the moment it is rented, while its container may still
    be pulling, stuck, or already exited -- `uptimeInSeconds` is what separates
    the three, and watching the wrong one cost $1.26 on 2026-08-22.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pod = api.pod_runtime(pod_id)
        uptime = ((pod.get("runtime") or {}).get("uptimeInSeconds")) or 0
        host_id = (pod.get("machine") or {}).get("podHostId")
        if uptime > 0 and host_id:
            return host_id
        time.sleep(poll_s)
    raise TimeoutError(f"pod {pod_id} container never started within {timeout_s}s")


def cmd_up(api: RunpodAPI, args: argparse.Namespace) -> int:
    sha, clean = git_image_tag()
    image = args.image or f"{IMAGE_REPO}:{sha}"
    if not clean and not args.image:
        print(f"WARNING: working tree is dirty; {image} is the *committed* tree. Results' "
              "provenance is the image, not what is on disk here.")

    account = api.account()
    balance = float(account.get("clientBalance") or 0.0)

    existing = find_pod(api.pods(), args.name)
    if existing and existing.get("desiredStatus") == "RUNNING":
        print(f"pod: reusing {existing['id']} ({args.name}), "
              f"${float(existing.get('costPerHr') or 0):.2f}/h")
        target = ssh_target(existing) or ssh_target(_wait_for_ssh(api, existing["id"], 300))
        print(f"\n  {ssh_command(*target)}\n")
        return 0
    if existing:
        print(f"pod {existing['id']} ({args.name}) exists but is {existing['desiredStatus']}.")
        if not args.resume_stopped:
            print("Pass --resume-stopped to restart it, or `down` to terminate it. Not "
                  "creating a second pod under the same name.")
            return 1
        api.resume_pod(existing["id"], args.gpu_count)
        pod = _wait_for_ssh(api, existing["id"], args.wait_seconds)
        print(f"\n  {ssh_command(*ssh_target(pod))}\n")
        return 0

    data_center, price = _pick_placement(api, args)
    if data_center is None and args.volume_gb:
        sys.exit("a network volume must live in a data center, and a COMMUNITY pod has "
                 "none. Pass --volume-gb 0 (the session needs no volume) or use "
                 "--cloud-type SECURE.")
    refusal = budget_verdict(balance, price, args.max_hours)
    if refusal:
        if args.dry_run:                     # a dry run reports the problem, it isn't blocked by it
            print(f"NOTE: {refusal}")
        elif not args.force:
            sys.exit(f"refusing to provision: {refusal}  (--force overrides)")
        else:
            print(f"WARNING (--force): {refusal}")

    volume = _ensure_volume(api, args, account, data_center)
    template = _ensure_template(api, args, account, image)

    started = datetime.now(UTC)
    deadline = deadline_iso(started, args.max_hours)
    env = {"DISTRAIN_DEADLINE_UTC": deadline}
    pubkey = read_pubkey(args.ssh_key) or account.get("pubKey") or ""
    if pubkey:
        env["PUBLIC_KEY"] = pubkey
    else:
        print("WARNING: no SSH public key found; the pod will boot without one and be "
              "unreachable. Pass --ssh-key.")

    plan = {
        "name": args.name, "image": image, "gpu": f"{args.gpu_count}x {args.gpu_type}",
        "cloud_type": args.cloud_type,
        "data_center": data_center or "(unpinned: RunPod places it)",
        "price_per_hr": price, "container_disk_gb":
        args.container_disk_gb, "pod_volume_gb": 0,
        "start_cmd": args.start_cmd or "(the image's own CMD)",
        "network_volume": (volume or {}).get("id"),
        "volume_mount_path": args.volume_mount_path if volume else None,
        "max_hours": args.max_hours, "deadline_utc": deadline,
        "estimated_ceiling_usd": round(price * args.max_hours, 2),
    }
    print("\nplan:")
    for key, value in plan.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        print("\n--dry-run: nothing rented.")
        return 0

    pod = api.create_pod(
        name=args.name,
        image_name=image,
        gpu_type_id=args.gpu_type,
        gpu_count=args.gpu_count,
        cloud_type=args.cloud_type,
        data_center_id=data_center,
        template_id=template["id"],
        docker_args=args.start_cmd,          # empty boots the image's own CMD
        container_disk_in_gb=args.container_disk_gb,
        volume_in_gb=0,                      # the API's 20 GB default would shadow /workspace
        volume_mount_path=args.volume_mount_path if volume else "/runpod-volume",
        network_volume_id=(volume or {}).get("id"),
        ports=POD_PORTS,
        env=env,
        support_public_ip=True,
        start_ssh=True,
    )
    pod_id = pod["id"]
    print(f"\npod: created {pod_id} -- billing from now, ${price:.2f}/h")

    try:                                     # teardown on exception: §9's kill-switch
        if args.ssh_proxy:
            host_id = _wait_for_container(api, pod_id, args.wait_seconds)
            session = {**plan, "pod_id": pod_id,
                       "started_utc": started.replace(microsecond=0).isoformat(),
                       "ssh": proxy_ssh_command(host_id), "pod_host_id": host_id,
                       "ip": None, "port": None}
            write_session(session, args.session_file)
            print(f"\n  {proxy_ssh_command(host_id)}\n")
            print(f"ceiling {args.max_hours:g} h -> terminate by {deadline} UTC "
                  f"(~${price * args.max_hours:.2f}). Start the kill-switch now:")
            print(f"  uv run --script {Path(__file__).relative_to(REPO_ROOT)} guard &")
            return 0
        ready = _wait_for_ssh(api, pod_id, args.wait_seconds)
    except BaseException as exc:             # includes KeyboardInterrupt
        if args.keep_on_error:
            print(f"\n{type(exc).__name__}: {exc}\npod {pod_id} LEFT RUNNING (--keep-on-error) "
                  f"-- it is billing at ${price:.2f}/h.")
            raise
        print(f"\n{type(exc).__name__}: {exc}\nterminating {pod_id} so it cannot bill quietly.")
        api.terminate_pod(pod_id)
        raise

    ip, port = ssh_target(ready)
    session = {**plan, "pod_id": pod_id, "started_utc": started.replace(microsecond=0).isoformat(),
               "ssh": ssh_command(ip, port), "ip": ip, "port": port}
    write_session(session, args.session_file)

    print(f"\n  {ssh_command(ip, port)}\n")
    print(f"ceiling {args.max_hours:g} h -> terminate by {deadline} UTC "
          f"(~${price * args.max_hours:.2f}). Start the kill-switch now:")
    print(f"  uv run --script {Path(__file__).relative_to(REPO_ROOT)} guard &")
    return 0


def cmd_guard(api: RunpodAPI, args: argparse.Namespace) -> int:
    """The cost kill-switch (docs/decisions.md §9): a hard wall-clock ceiling that
    terminates the pod, plus a balance watch, so an unattended run cannot outlive
    either. Terminating loses the container disk -- pull artifacts before the
    deadline, or extend it deliberately."""
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
          f"balance floor ${args.min_balance:.0f}, poll {args.poll_seconds}s")
    while True:
        now = datetime.now(UTC)
        pod = api.pod(pod_id)
        if not pod or pod.get("desiredStatus") == "TERMINATED":
            print(f"[{now:%H:%M:%S}] pod {pod_id} is gone; nothing to guard.")
            return 0

        cost = float(pod.get("costPerHr") or 0)
        balance = float(api.account().get("clientBalance") or 0)
        left = (deadline_dt - now).total_seconds() / 3600
        print(f"[{now:%H:%M:%S}] ${balance:.2f} left, ${cost:.2f}/h, runway "
              f"{runway_hours(balance, cost):.1f} h, ceiling in {left:.2f} h", flush=True)

        if now >= deadline_dt:
            print(f"CEILING REACHED -- terminating {pod_id}.")
            api.terminate_pod(pod_id)
            return 2
        if balance < args.min_balance:
            print(f"LOW BALANCE: ${balance:.2f} < ${args.min_balance:.2f}. RunPod terminates "
                  "pods (and their container disks) at zero -- pull artifacts now.")
            if args.terminate_on_low_balance:
                print(f"terminating {pod_id} (--terminate-on-low-balance).")
                api.terminate_pod(pod_id)
                return 3
        if runway_hours(balance, cost) < left:
            print(f"NOTE: credit runs out {left - runway_hours(balance, cost):.1f} h before "
                  "the ceiling.")
        time.sleep(args.poll_seconds)


def cmd_ssh(api: RunpodAPI, args: argparse.Namespace) -> int:
    session = read_session(args.session_file)
    pod_id = args.pod_id or session.get("pod_id")
    pod = api.pod(pod_id) if pod_id else find_pod(api.pods(), args.name)
    if not pod:
        sys.exit("no pod found (try `status`)")
    target = ssh_target(pod)
    if not target:
        sys.exit(f"pod {pod['id']} has no public port 22 yet")
    command = ssh_command(*target)
    if args.exec:
        return subprocess.call(command.split() + args.rest)
    print(command)
    return 0


def cmd_down(api: RunpodAPI, args: argparse.Namespace) -> int:
    session = read_session(args.session_file)
    pod_id = args.pod_id or session.get("pod_id")
    pod = api.pod(pod_id) if pod_id else find_pod(api.pods(), args.name)

    if pod:
        print(f"terminating {pod['id']} ({pod.get('name')}) -- the container disk goes with it.")
        if not args.yes and input("type the pod id to confirm: ").strip() != pod["id"]:
            return 1
        api.terminate_pod(pod["id"])
        print("terminated.")
    else:
        print("no pod to terminate.")

    if args.delete_volume:
        volume = find_volume(api.account().get("networkVolumes") or [], args.volume_name, 0)
        if volume:
            print(f"deleting volume {volume['id']} ({volume['name']}, {volume['size']} GB)")
            api.delete_volume(volume["id"])
        else:
            print(f"no volume named {args.volume_name}.")

    if args.session_file.exists():
        args.session_file.unlink()

    print()
    return cmd_verify(api, args)      # teardown ends with the proof, not with a claim


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session-file", type=Path, default=SESSION_FILE)
    parser.add_argument("--name", default=None,
                        help="pod name; defaults to distrain-<gpu>x<count>")
    parser.add_argument("--gpu-type", default=GPU_TYPE)
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT)
    parser.add_argument("--skip-capacity-check", action="store_true",
                        help="do not gate on the lowestPrice availability query; let the "
                             "deploy call place the pod (the precheck has false negatives)")
    parser.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"],
                        default="SECURE",
                        help="COMMUNITY is cheaper and often the only stock for "
                             "consumer GPUs, but has no data center id, so it "
                             "cannot carry a network volume")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="balance, pods, volumes, registry credentials")

    verify = subparsers.add_parser(
        "verify", aliases=["check"],
        help="is anything still costing money? exits non-zero if so")
    verify.add_argument("--strict", action="store_true",
                        help="network volumes count as billing too (end of project, not of a "
                             "session)")
    verify.add_argument("--settle-seconds", type=int, default=45,
                        help="how long to let billing catch up before believing a spend figure "
                             "no pod explains; 0 to answer immediately")

    avail = subparsers.add_parser("avail", help="per-data-center capacity for the GPU type")
    avail.add_argument("--data-centers", nargs="*", default=None)

    up = subparsers.add_parser("up", help="ensure volume + pod, boot the image, wait for SSH")
    up.add_argument("--max-hours", type=float, default=8.0,
                    help="wall-clock ceiling the budget check and `guard` enforce")
    up.add_argument("--image", default=None, help=f"default: {IMAGE_REPO}:<git short sha>")
    up.add_argument("--data-center", default=None, help="pin one; default: cheapest with stock")
    up.add_argument("--data-centers", nargs="*", default=None, help="restrict the search")
    up.add_argument("--container-disk-gb", type=int, default=CONTAINER_DISK_GB)
    up.add_argument("--volume-gb", type=int, default=100,
                    help="network volume size; 0 disables it (it bills until deleted)")
    up.add_argument("--volume-name", default="distrain")
    up.add_argument("--volume-mount-path", default=VOLUME_MOUNT_PATH)
    up.add_argument("--ssh-key", type=Path, default=None, help="public key for the pod")
    up.add_argument("--registry-auth-name", default=REGISTRY_AUTH_NAME,
                    help="registry credential to pull with; '' for a public image. The "
                         "credential is handed to whichever registry the image names, so "
                         "the wrong one fails the pull rather than being ignored.")
    up.add_argument("--ghcr-user", default="adamdivak")
    up.add_argument("--ghcr-token", default=os.environ.get("GHCR_TOKEN"),
                    help="PAT with read:packages; only needed to create the credential once")
    up.add_argument("--wait-seconds", type=int, default=600)
    up.add_argument("--resume-stopped", action="store_true",
                    help="restart a stopped pod of the same name instead of failing")
    up.add_argument("--ssh-proxy", action="store_true",
                    help="reach the pod through ssh.runpod.io instead of a public port. "
                         "Community hosts expose no public port, so this is the only way "
                         "onto one; it needs the container running, not sshd.")
    up.add_argument("--start-cmd", default=START_CMD,
                    help="command the container runs; '' boots the image's own CMD, which "
                         "is what a stock image (runpod/pytorch, ...) needs to start its "
                         "own sshd. Default assumes our image's pod-entry.sh.")
    up.add_argument("--keep-on-error", action="store_true",
                    help="do NOT terminate the pod if provisioning fails (debugging only)")
    up.add_argument("--force", action="store_true", help="provision despite the budget check")
    up.add_argument("--dry-run", action="store_true", help="print the plan, rent nothing")

    guard = subparsers.add_parser("guard", help="wall-clock ceiling + balance watch")
    guard.add_argument("--pod-id", default=None)
    guard.add_argument("--max-hours", type=float, default=None,
                       help="restart the ceiling from now; default: the deadline from `up`")
    guard.add_argument("--deadline", default=None, help="explicit ISO-8601 UTC deadline")
    guard.add_argument("--min-balance", type=float, default=20.0)
    guard.add_argument("--terminate-on-low-balance", action="store_true")
    guard.add_argument("--poll-seconds", type=int, default=60)

    ssh = subparsers.add_parser("ssh", help="print (or run) the ssh command")
    ssh.add_argument("--pod-id", default=None)
    ssh.add_argument("--exec", action="store_true", help="run ssh instead of printing it")
    ssh.add_argument("rest", nargs="*", help="command to run on the pod (with --exec)")

    down = subparsers.add_parser("down", help="terminate the pod (optionally the volume)")
    down.add_argument("--pod-id", default=None)
    down.add_argument("--delete-volume", action="store_true")
    down.add_argument("--volume-name", default="distrain")
    down.add_argument("-y", "--yes", action="store_true", help="skip the id confirmation")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.name:
        short = args.gpu_type.split()[-1].split("-")[0].lower()   # A100-SXM4-80GB -> a100
        args.name = f"distrain-{short}x{args.gpu_count}"
    runpod.api_key = load_api_key()

    commands = {"status": cmd_status, "verify": cmd_verify, "check": cmd_verify,
                "avail": cmd_avail, "up": cmd_up, "guard": cmd_guard,
                "ssh": cmd_ssh, "down": cmd_down}
    try:
        return commands[args.command](RunpodAPI(), args)
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
