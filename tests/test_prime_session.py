"""Offline checks of scripts/prime_session.py against a fake Prime Intellect API.

No network, no credentials, nothing rented. What is exercised is exactly what
costs money or invalidates a result when it is wrong: the socket is pinned to
SXM4 so a PCIe box cannot masquerade as the anchor's hardware, a pod without a
custom template is refused rather than booted on Prime Intellect's own image, an
existing pod is reused rather than duplicated, a failure after creation tears the
pod down, and the wall-clock ceiling terminates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "prime_session.py"


def _load_script():
    """The script is a PEP 723 stdlib-only script, so it imports as-is."""
    spec = importlib.util.spec_from_file_location("prime_session", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prime_session"] = module
    spec.loader.exec_module(module)
    return module


ps = _load_script()


def offer_fixture(**overrides) -> dict:
    offer = {
        "cloudId": "gpu_8x_a100_80gb_sxm4", "gpuType": "A100_80GB", "socket": "SXM4",
        "provider": "lambdalabs", "region": "united_states", "dataCenter": "us-east-1",
        "country": "US", "gpuCount": 8, "stockStatus": "Available",
        "prices": {"onDemand": 22.32},
    }
    offer.update(overrides)
    return offer


def pod_fixture(**overrides) -> dict:
    pod = {
        "id": "pod123", "name": "distrain-a100x8", "status": "ACTIVE",
        "priceHr": 22.32, "gpuCount": 8, "gpuName": "A100_80GB", "socket": "SXM4",
        "providerType": "lambdalabs", "sshConnection": "ubuntu@185.216.23.121 -p 2222",
        "ip": "185.216.23.121",
    }
    pod.update(overrides)
    return pod


class FakeAPI:
    def __init__(self, pods=None, balance=500.0, offers=None, ssh_keys=None,
                 registry=None, fail_create=False):
        self._pods = list(pods or [])
        self._balance = balance
        self._offers = offers if offers is not None else [offer_fixture()]
        self._ssh_keys = list(ssh_keys or [])
        self._registry = list(registry or [])
        self._fail_create = fail_create
        self.created_pods: list[dict] = []
        self.terminated: list[str] = []
        self.created_keys: list[dict] = []

    def wallet(self):
        return {"balance_usd": self._balance}

    def whoami(self):
        return {"data": {"email": "test@example.com"}}

    def registry_credentials(self):
        return list(self._registry)

    def availability(self, gpu_type=None, gpu_count=None):
        return {"A100_80GB": list(self._offers)}

    def pods(self):
        return list(self._pods)

    def pod(self, pod_id):
        for pod in self._pods:
            if pod["id"] == pod_id:
                return pod
        return None

    def create_pod(self, payload):
        if self._fail_create:
            raise RuntimeError("create failed")
        self.created_pods.append(payload)
        # Reachable on the first poll, so `_wait_for_ssh` returns without sleeping.
        self._pods.append(pod_fixture(id="new1", name=payload["pod"]["name"]))
        return {"id": "new1"}

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        for pod in self._pods:
            if pod["id"] == pod_id:
                pod["status"] = "TERMINATED"

    def ssh_keys(self):
        return list(self._ssh_keys)

    def create_ssh_key(self, name, public_key):
        self.created_keys.append({"name": name, "publicKey": public_key})
        return {"id": "key1"}


def up_args(tmp_path, **overrides):
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAATEST user@aurora\n")
    argv = ["--session-file", str(tmp_path / "session.json"), "up",
            "--image", "ghcr.io/adamdivak/distrain:abc1234",
            "--ssh-key", str(tmp_path / "id.pub"),
            "--template-id", "tmpl-abc"]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv += [flag]
        elif value is False:
            continue
        else:
            argv += [flag, str(value)]
    args = ps.build_parser().parse_args(argv)
    args.name = "distrain-a100x8"
    return args


class TestPureHelpers:
    def test_parses_the_ssh_connection_shapes_the_upstreams_return(self):
        assert ps.parse_ssh_connection("ubuntu@1.2.3.4") == ("ubuntu", "1.2.3.4", 22)
        assert ps.parse_ssh_connection("ubuntu@1.2.3.4 -p 2222") == ("ubuntu", "1.2.3.4", 2222)
        assert ps.parse_ssh_connection("root@1.2.3.4:2200") == ("root", "1.2.3.4", 2200)
        assert ps.parse_ssh_connection(["ubuntu@1.2.3.4", "-p", "22"]) == ("ubuntu", "1.2.3.4", 22)
        assert ps.parse_ssh_connection(None) == (None, None, 22)

    def test_ssh_target_falls_back_to_the_ip_field(self):
        assert ps.ssh_target(pod_fixture()) == ("ubuntu", "185.216.23.121", 2222)
        assert ps.ssh_target(pod_fixture(sshConnection=None)) == ("root", "185.216.23.121", 22)
        assert ps.ssh_target(pod_fixture(sshConnection=None, ip=None)) is None
        assert ps.ssh_command("ubuntu", "1.2.3.4", 22) == "ssh -p 22 ubuntu@1.2.3.4"

    def test_budget_verdict_refuses_what_the_balance_cannot_finish(self):
        assert ps.budget_verdict(500.0, 22.32, 4) is None
        message = ps.budget_verdict(20.0, 22.32, 4)
        assert message and "0.9 h" in message      # the ceiling $20 actually covers

    def test_runway_and_deadline_arithmetic(self):
        assert ps.runway_hours(100.0, 12.5) == 8.0
        assert ps.runway_hours(100.0, 0) == float("inf")
        started = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
        assert ps.deadline_iso(started, 4) == "2026-08-19T16:00:00+00:00"

    def test_in_stock_treats_low_as_rentable_and_unavailable_as_not(self):
        assert ps.in_stock(offer_fixture(stockStatus="Available"))
        assert ps.in_stock(offer_fixture(stockStatus="Low"))
        assert not ps.in_stock(offer_fixture(stockStatus="Unavailable"))
        assert not ps.in_stock(offer_fixture(stockStatus=None))

    def test_matching_offers_pins_the_socket(self):
        """A PCIe A100 is not the SXM4 anchor's hardware. Catching this here is
        the whole point: both are 'A100_80GB x8' and priced alike, and a run on
        the wrong one produces a number that looks fine and compares to nothing.
        """
        availability = {"A100_80GB": [
            offer_fixture(socket="PCIe", prices={"onDemand": 15.92}),
            offer_fixture(socket="SXM4", prices={"onDemand": 22.32}),
        ]}
        sxm = ps.matching_offers(availability, "A100_80GB", 8, socket="SXM4")
        assert [o["socket"] for o in sxm] == ["SXM4"]
        assert len(ps.matching_offers(availability, "A100_80GB", 8, socket="any")) == 2

    def test_matching_offers_sorts_by_price_and_drops_the_unrentable(self):
        availability = {"A100_80GB": [
            offer_fixture(provider="vultr", prices={"onDemand": 22.40}),
            offer_fixture(provider="lambdalabs", prices={"onDemand": 22.32}),
            offer_fixture(provider="hyperstack", stockStatus="Unavailable",
                          prices={"onDemand": 11.20}),
            offer_fixture(provider="latitude", prices={"onDemand": None}),
            offer_fixture(provider="oblivus", gpuCount=4),
        ]}
        offers = ps.matching_offers(availability, "A100_80GB", 8, socket="SXM4")
        assert [o["provider"] for o in offers] == ["lambdalabs", "vultr"]

    def test_find_pod_ignores_terminated_namesakes(self):
        pods = [pod_fixture(id="dead", status="TERMINATED"), pod_fixture(id="live")]
        assert ps.find_pod(pods, "distrain-a100x8")["id"] == "live"
        assert ps.find_pod([pod_fixture(status="TERMINATED")], "distrain-a100x8") is None


class TestUp:
    def test_creates_the_pod_with_the_load_bearing_settings(self, tmp_path):
        api = FakeAPI()
        assert ps.cmd_up(api, up_args(tmp_path, max_hours=4)) == 0

        (created,) = api.created_pods
        pod = created["pod"]
        assert created["provider"]["type"] == "lambdalabs"
        assert pod["socket"] == "SXM4"                  # the anchor's hardware
        assert pod["image"] == "custom_template"        # not PI's own environment
        assert pod["customTemplateId"] == "tmpl-abc"
        assert pod["gpuCount"] == 8
        assert pod["autoRestart"] is False              # a restart would bill past the ceiling
        assert pod["maxPrice"] == pytest.approx(22.32 * 1.05)
        assert pod["sshKeyId"] == "key1"
        deadline = [e for e in pod["envVars"] if e["key"] == "DISTRAIN_DEADLINE_UTC"]
        assert len(deadline) == 1

    def test_refuses_to_boot_prime_intellects_own_image(self, tmp_path, capsys):
        """Without a template the pod runs their ubuntu and none of our code, so
        anything measured on it is unreportable. Refuse loudly instead."""
        api = FakeAPI()
        args = up_args(tmp_path)
        args.template_id = None
        assert ps.cmd_up(api, args) == 2
        assert not api.created_pods
        assert "--template-id" in capsys.readouterr().out

    def test_allow_stock_image_is_an_explicit_opt_out(self, tmp_path):
        api = FakeAPI()
        args = up_args(tmp_path)
        args.template_id = None
        args.allow_stock_image = True
        assert ps.cmd_up(api, args) == 0
        (created,) = api.created_pods
        assert created["pod"]["image"] == "ubuntu_22_cuda_12"
        assert "customTemplateId" not in created["pod"]

    def test_reuses_a_running_pod_instead_of_renting_a_second(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture()])
        assert ps.cmd_up(api, up_args(tmp_path)) == 0
        assert not api.created_pods

    def test_refuses_when_the_balance_cannot_fund_the_ceiling(self, tmp_path):
        """$20 against a $22.32/h box: the exact case on 2026-08-19."""
        api = FakeAPI(balance=20.0)
        with pytest.raises(SystemExit):
            ps.cmd_up(api, up_args(tmp_path, max_hours=4))
        assert not api.created_pods

    def test_force_overrides_the_budget_refusal(self, tmp_path):
        api = FakeAPI(balance=20.0)
        assert ps.cmd_up(api, up_args(tmp_path, max_hours=4, force=True)) == 0
        assert api.created_pods

    def test_a_dry_run_still_prints_the_plan_it_cannot_afford(self, tmp_path, capsys):
        api = FakeAPI(balance=20.0)
        assert ps.cmd_up(api, up_args(tmp_path, max_hours=4, dry_run=True)) == 0
        out = capsys.readouterr().out
        assert "NOTE:" in out and "nothing rented" in out
        assert not api.created_pods

    def test_reports_no_capacity_without_renting_a_different_shape(self, tmp_path):
        api = FakeAPI(offers=[offer_fixture(socket="PCIe")])
        assert ps.cmd_up(api, up_args(tmp_path)) == 1
        assert not api.created_pods

    def test_terminates_the_pod_when_provisioning_fails_after_creation(self, tmp_path,
                                                                       monkeypatch):
        api = FakeAPI()
        monkeypatch.setattr(ps, "_wait_for_ssh",
                            lambda *a, **k: (_ for _ in ()).throw(TimeoutError("no ssh")))
        with pytest.raises(TimeoutError):
            ps.cmd_up(api, up_args(tmp_path))
        assert api.terminated == ["new1"]

    def test_keep_on_error_leaves_the_pod_but_still_raises(self, tmp_path, monkeypatch):
        api = FakeAPI()
        monkeypatch.setattr(ps, "_wait_for_ssh",
                            lambda *a, **k: (_ for _ in ()).throw(TimeoutError("no ssh")))
        with pytest.raises(TimeoutError):
            ps.cmd_up(api, up_args(tmp_path, keep_on_error=True))
        assert api.terminated == []

    def test_teardown_failure_does_not_mask_the_original_error(self, tmp_path, monkeypatch,
                                                               capsys):
        api = FakeAPI()
        monkeypatch.setattr(ps, "_wait_for_ssh",
                            lambda *a, **k: (_ for _ in ()).throw(TimeoutError("no ssh")))
        monkeypatch.setattr(api, "terminate_pod",
                            lambda pod_id: (_ for _ in ()).throw(RuntimeError("API down")))
        with pytest.raises(TimeoutError):
            ps.cmd_up(api, up_args(tmp_path))
        assert "BY HAND" in capsys.readouterr().out

    def test_writes_the_session_file(self, tmp_path):
        api = FakeAPI()
        args = up_args(tmp_path)
        ps.cmd_up(api, args)
        session = json.loads((tmp_path / "session.json").read_text())
        assert session["pod_id"] == "new1"
        assert session["price_per_hr"] == 22.32
        assert session["deadline_utc"]


class TestGuard:
    def _guard_args(self, tmp_path, **overrides):
        argv = ["--session-file", str(tmp_path / "session.json"), "guard"]
        for key, value in overrides.items():
            argv += ["--" + key.replace("_", "-"), str(value)]
        return ps.build_parser().parse_args(argv)

    def test_terminates_at_the_ceiling(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture()])
        past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
        args = self._guard_args(tmp_path, pod_id="pod123", deadline=past)
        assert ps.cmd_guard(api, args) == 0
        assert api.terminated == ["pod123"]

    def test_terminates_when_the_balance_hits_the_floor(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture()], balance=1.0)
        future = (datetime.now(UTC) + timedelta(hours=4)).isoformat()
        args = self._guard_args(tmp_path, pod_id="pod123", deadline=future, min_balance=2.0)
        assert ps.cmd_guard(api, args) == 0
        assert api.terminated == ["pod123"]

    def test_stops_quietly_when_the_pod_is_already_gone(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture(status="TERMINATED")])
        future = (datetime.now(UTC) + timedelta(hours=4)).isoformat()
        args = self._guard_args(tmp_path, pod_id="pod123", deadline=future)
        assert ps.cmd_guard(api, args) == 0
        assert api.terminated == []


class TestAvailAndVerify:
    def _args(self, command, **overrides):
        argv = [command]
        args = ps.build_parser().parse_args(argv)
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_avail_exits_zero_only_when_something_is_rentable(self, capsys):
        api = FakeAPI()
        assert ps.cmd_avail(api, self._args("avail")) == 0
        assert "lambdalabs" in capsys.readouterr().out

        empty = FakeAPI(offers=[offer_fixture(stockStatus="Unavailable")])
        assert ps.cmd_avail(empty, self._args("avail")) == 1

    def test_avail_names_the_socket_it_refused_to_count(self, capsys):
        api = FakeAPI(offers=[offer_fixture(socket="PCIe")])
        assert ps.cmd_avail(api, self._args("avail")) == 1
        assert "PCIe" in capsys.readouterr().out

    def test_verify_is_strict_about_anything_still_billing(self):
        api = FakeAPI(pods=[pod_fixture()])
        assert ps.cmd_verify(api, self._args("verify", strict=True)) == 1
        assert ps.cmd_verify(api, self._args("verify", strict=False)) == 0

    def test_verify_is_clean_when_everything_is_terminated(self, capsys):
        api = FakeAPI(pods=[pod_fixture(status="TERMINATED")])
        assert ps.cmd_verify(api, self._args("verify", strict=True)) == 0
        assert "nothing is billing" in capsys.readouterr().out

    def test_status_warns_when_no_registry_credential_exists(self, capsys):
        api = FakeAPI()
        assert ps.cmd_status(api, self._args("status")) == 0
        assert "no private-registry credential" in capsys.readouterr().out


class TestDown:
    def test_terminates_and_clears_the_session(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture()])
        session = tmp_path / "session.json"
        session.write_text(json.dumps({"pod_id": "pod123"}))
        args = ps.build_parser().parse_args(["--session-file", str(session), "down"])
        assert ps.cmd_down(api, args) == 0
        assert api.terminated == ["pod123"]
        assert not session.exists()

    def test_is_a_noop_when_nothing_is_running(self, tmp_path):
        api = FakeAPI(pods=[])
        args = ps.build_parser().parse_args(
            ["--session-file", str(tmp_path / "session.json"), "down"])
        assert ps.cmd_down(api, args) == 0
        assert api.terminated == []
