"""Offline checks of scripts/runpod_session.py against a fake RunPod API.

No network, no credentials, nothing rented. What is exercised is exactly what
costs money when it is wrong: the pod is created with a 0 GB pod volume and the
right start command, an existing pod is reused rather than duplicated, a failure
after creation tears the pod down, and the wall-clock ceiling terminates.

The script pulls the `runpod` SDK from its own inline (PEP 723) dependency block,
so the SDK is not in this project's environment -- a stub stands in for the
import, and every call goes through the injected API seam instead.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "runpod_session.py"


def _load_script():
    """Import the script with the runpod SDK stubbed out."""
    sdk = types.ModuleType("runpod")
    sdk.api_key = None
    sdk.error = types.SimpleNamespace(QueryError=RuntimeError, AuthenticationError=RuntimeError)
    api_pkg = types.ModuleType("runpod.api")
    graphql = types.ModuleType("runpod.api.graphql")
    graphql.run_graphql_query = lambda query: pytest.fail("script hit the live API")
    api_pkg.graphql = graphql
    sdk.api = api_pkg
    sys.modules.update({"runpod": sdk, "runpod.api": api_pkg, "runpod.api.graphql": graphql})

    spec = importlib.util.spec_from_file_location("runpod_session", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rps = _load_script()


def pod_fixture(**overrides) -> dict:
    pod = {
        "id": "pod123", "name": "distrain-a100x8", "desiredStatus": "RUNNING",
        "costPerHr": 12.72, "gpuCount": 8, "imageName": "ghcr.io/adamdivak/distrain:abc1234",
        "uptimeSeconds": 60, "machine": {"gpuDisplayName": "A100 SXM"},
        "runtime": {"ports": [
            {"privatePort": 22, "publicPort": 31722, "ip": "185.216.23.121",
             "isIpPublic": True, "type": "tcp"},
        ]},
    }
    pod.update(overrides)
    return pod


class FakeAPI:
    """Records what the script would have done to the account."""

    def __init__(self, pods=None, volumes=None, balance=500.0, templates=None,
                 creds=None, price=1.59, ready_pod=None, endpoints=None,
                 reported_spend=None):
        self._endpoints = list(endpoints or [])
        self.reported_spend = reported_spend
        self._pods = list(pods or [])
        self._volumes = list(volumes or [])
        self._templates = list(templates or [])
        self._creds = creds if creds is not None else [{"id": "auth1", "name": "GitHub packages"}]
        self._balance = balance
        self._price = price
        self._ready_pod = ready_pod
        self.created_pods, self.created_volumes, self.terminated, self.resumed = [], [], [], []

    def account(self):
        spend = sum(float(p.get("costPerHr") or 0) for p in self._pods
                    if p.get("desiredStatus") == "RUNNING")
        if isinstance(self.reported_spend, list):      # one value per account() call
            spend = self.reported_spend.pop(0) if self.reported_spend else 0.0
        elif self.reported_spend is not None:
            spend = self.reported_spend
        return {"id": "u1", "pubKey": "ssh-ed25519 ACCOUNT", "clientBalance": self._balance,
                "currentSpendPerHr": spend,
                "networkVolumes": self._volumes,
                "containerRegistryCreds": self._creds, "podTemplates": self._templates}

    def pods(self):
        return self._pods

    def endpoints(self):
        return self._endpoints

    def pod(self, pod_id):
        if self._ready_pod is not None:
            return self._ready_pod
        return next((p for p in self._pods if p["id"] == pod_id), None)

    def create_pod(self, **kwargs):
        self.created_pods.append(kwargs)
        created = pod_fixture(id="new1", name=kwargs["name"])
        self._pods.append(created)
        if self._ready_pod is None:
            self._ready_pod = created
        return {"id": "new1"}

    def resume_pod(self, pod_id, gpu_count):
        self.resumed.append((pod_id, gpu_count))
        return {"id": pod_id}

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        self._pods = [p for p in self._pods if p["id"] != pod_id]
        self._ready_pod = None

    def create_template(self, **kwargs):
        self._templates.append({"id": "tpl1", **kwargs})
        return {"id": "tpl1", "name": kwargs["name"], "imageName": kwargs["image_name"]}

    def create_registry_auth(self, name, username, password):
        auth = {"id": "auth-new", "name": name}
        self._creds.append(auth)
        return auth

    def create_volume(self, name, size_gb, data_center_id):
        volume = {"id": "vol1", "name": name, "size": size_gb, "dataCenterId": data_center_id}
        self.created_volumes.append(volume)
        self._volumes.append(volume)
        return volume

    def delete_volume(self, volume_id):
        self._volumes = [v for v in self._volumes if v["id"] != volume_id]

    def gpu_price(self, gpu_type, gpu_count, data_center_id=None, secure=None):
        # `price` is the per-GPU secure rate (1.59 x 8 = the $12.72/h an 8x A100
        # pod really bills); lowestPrice is the community rate, and only its
        # presence means capacity.
        #
        # `secure=False` asks for community stock, which real hosts report only
        # on the unscoped query -- they carry no data center id, so pairing it
        # with one is always empty. Modelled here so a per-DC community lookup
        # cannot silently look available in tests when it never is in practice.
        if secure is False and data_center_id is not None:
            available = False
        else:
            available = data_center_id in (None, "US-KS-2", "EU-RO-1")
        return {"id": gpu_type, "maxGpuCount": 8,
                "securePrice": self._price, "communityPrice": self._price * 0.5,
                "lowestPrice": {
                    "uninterruptablePrice": self._price * 0.5 if available else None,
                    "minimumBidPrice": self._price * 0.5 if available else None,
                    "stockStatus": "High" if available else None}}


def up_args(tmp_path, **overrides):
    argv = ["--session-file", str(tmp_path / "session.json"), "up",
            "--image", "ghcr.io/adamdivak/distrain:abc1234",
            "--ssh-key", str(tmp_path / "id.pub"), "--data-centers", "US-KS-2"]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAATEST user@aurora\n")
    args = rps.build_parser().parse_args(argv)
    args.name = "distrain-a100x8"
    return args


class TestPureHelpers:
    def test_ssh_target_picks_the_public_port_22_mapping(self):
        assert rps.ssh_target(pod_fixture()) == ("185.216.23.121", 31722)
        assert rps.ssh_command("1.2.3.4", 22) == "ssh -p 22 root@1.2.3.4"

    def test_ssh_target_is_none_until_the_port_maps(self):
        assert rps.ssh_target(pod_fixture(runtime=None)) is None
        assert rps.ssh_target(pod_fixture(runtime={"ports": [
            {"privatePort": 22, "publicPort": 1, "ip": "10.0.0.1", "isIpPublic": False}]})) is None

    def test_budget_verdict_refuses_what_the_balance_cannot_finish(self):
        assert rps.budget_verdict(500.0, 12.72, 8) is None
        message = rps.budget_verdict(19.48, 12.72, 8)
        assert message and "1.5 h" in message           # the ceiling the balance does cover

    def test_runway_and_deadline_arithmetic(self):
        assert rps.runway_hours(100.0, 12.5) == 8.0
        assert rps.runway_hours(100.0, 0) == float("inf")
        started = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
        assert rps.deadline_iso(started, 8) == "2026-08-18T20:00:00+00:00"

    def test_secure_price_is_per_gpu_times_count(self):
        """Measured on a real pod: a 3090 billed $0.50/h while `lowestPrice`
        said $0.22 (that is the community rate). SECURE pods bill securePrice."""
        info = {"securePrice": 1.59, "communityPrice": 1.39,
                "lowestPrice": {"uninterruptablePrice": 1.39}}
        assert rps.secure_price(info, 8) == pytest.approx(12.72)
        assert rps.secure_price({}, 8) == 0.0

    def test_lookups_by_name_and_size(self):
        pods = [pod_fixture(id="a", name="other"), pod_fixture(id="b", name="distrain-a100x8")]
        assert rps.find_pod(pods, "distrain-a100x8")["id"] == "b"
        assert rps.find_pod(pods, "nope") is None
        volumes = [{"name": "distrain", "size": 50}, {"name": "distrain", "size": 200}]
        assert rps.find_volume(volumes, "distrain", 100)["size"] == 200
        assert rps.find_volume(volumes, "distrain", 500) is None


class TestUp:
    def test_creates_the_pod_with_the_load_bearing_settings(self, tmp_path):
        api = FakeAPI()
        assert rps.cmd_up(api, up_args(tmp_path, volume_gb=0, max_hours=8)) == 0

        (created,) = api.created_pods
        # a non-zero pod volume would mount at /workspace and shadow the baked code
        assert created["volume_in_gb"] == 0
        # an empty start command boots the image CMD, which exits -> boot loop
        assert created["docker_args"] == rps.START_CMD
        assert created["ports"] == "22/tcp"
        assert created["gpu_count"] == 8 and created["cloud_type"] == "SECURE"
        assert created["env"]["PUBLIC_KEY"].startswith("ssh-ed25519")
        assert created["env"]["DISTRAIN_DEADLINE_UTC"].endswith("+00:00")
        assert created["network_volume_id"] is None

        session = json.loads((tmp_path / "session.json").read_text())
        assert session["pod_id"] == "new1" and session["ssh"].startswith("ssh -p 31722 root@")
        assert session["max_hours"] == 8 and session["price_per_hr"] == 12.72

    def test_reuses_a_running_pod_instead_of_renting_a_second(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture()])
        assert rps.cmd_up(api, up_args(tmp_path)) == 0
        assert api.created_pods == []

    def test_refuses_to_shadow_a_stopped_pod_of_the_same_name(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture(desiredStatus="EXITED", runtime=None)])
        assert rps.cmd_up(api, up_args(tmp_path)) == 1
        assert api.created_pods == [] and api.resumed == []

        api = FakeAPI(pods=[pod_fixture(desiredStatus="EXITED")])
        assert rps.cmd_up(api, up_args(tmp_path, resume_stopped=True)) == 0
        assert api.resumed == [("pod123", 8)] and api.created_pods == []

    def test_refuses_when_the_balance_cannot_fund_the_ceiling(self, tmp_path):
        api = FakeAPI(balance=19.48)
        with pytest.raises(SystemExit, match="refusing to provision"):
            rps.cmd_up(api, up_args(tmp_path, volume_gb=0, max_hours=8))
        assert api.created_pods == []

        assert rps.cmd_up(api, up_args(tmp_path, volume_gb=0, max_hours=1, force=True)) == 0

    def test_a_dry_run_still_prints_the_plan_it_cannot_afford(self, tmp_path, capsys):
        api = FakeAPI(balance=19.48)
        assert rps.cmd_up(api, up_args(tmp_path, volume_gb=0, max_hours=8, dry_run=True)) == 0
        out = capsys.readouterr().out
        assert "cannot fund" in out and "estimated_ceiling_usd" in out

    def test_dry_run_rents_nothing(self, tmp_path):
        api = FakeAPI()
        assert rps.cmd_up(api, up_args(tmp_path, dry_run=True, volume_gb=100)) == 0
        assert api.created_pods == [] and api.created_volumes == []
        assert not (tmp_path / "session.json").exists()

    def test_terminates_the_pod_when_provisioning_fails_after_creation(self, tmp_path, monkeypatch):
        api = FakeAPI()
        monkeypatch.setattr(rps, "_wait_for_ssh",
                            lambda *a, **k: (_ for _ in ()).throw(TimeoutError("no port 22")))
        with pytest.raises(TimeoutError):
            rps.cmd_up(api, up_args(tmp_path, volume_gb=0))
        assert api.terminated == ["new1"]                      # nothing left billing

    def test_keep_on_error_leaves_the_pod_but_still_raises(self, tmp_path, monkeypatch):
        api = FakeAPI()
        monkeypatch.setattr(rps, "_wait_for_ssh",
                            lambda *a, **k: (_ for _ in ()).throw(TimeoutError("no port 22")))
        with pytest.raises(TimeoutError):
            rps.cmd_up(api, up_args(tmp_path, volume_gb=0, keep_on_error=True))
        assert api.terminated == []

    def test_no_data_center_with_capacity_is_a_refusal(self, tmp_path):
        api = FakeAPI()
        args = up_args(tmp_path, volume_gb=0)
        args.data_centers = ["US-TX-3"]                        # FakeAPI reports no stock there
        with pytest.raises(SystemExit, match="no capacity"):
            rps.cmd_up(api, args)
        assert api.created_pods == []


class TestVolume:
    def test_creates_one_when_missing_and_attaches_it_at_data(self, tmp_path):
        api = FakeAPI()
        assert rps.cmd_up(api, up_args(tmp_path, volume_gb=100)) == 0
        assert api.created_volumes == [
            {"id": "vol1", "name": "distrain", "size": 100, "dataCenterId": "US-KS-2"}]
        (created,) = api.created_pods
        assert created["network_volume_id"] == "vol1"
        assert created["volume_mount_path"] == "/data"
        assert created["volume_in_gb"] == 0                    # still no pod volume

    def test_reuses_an_existing_volume_of_sufficient_size(self, tmp_path):
        volume = {"id": "vol9", "name": "distrain", "size": 200, "dataCenterId": "US-KS-2"}
        api = FakeAPI(volumes=[volume])
        assert rps.cmd_up(api, up_args(tmp_path, volume_gb=100)) == 0
        assert api.created_volumes == []
        assert api.created_pods[0]["network_volume_id"] == "vol9"

    def test_a_volume_in_another_data_center_stops_the_session(self, tmp_path):
        api = FakeAPI(volumes=[{"id": "vol9", "name": "distrain", "size": 200,
                                "dataCenterId": "EU-RO-1"}])
        with pytest.raises(SystemExit, match="EU-RO-1"):
            rps.cmd_up(api, up_args(tmp_path, volume_gb=100))
        assert api.created_pods == []

    def test_mounting_at_workspace_is_refused(self, tmp_path):
        api = FakeAPI()
        with pytest.raises(SystemExit, match="/workspace"):
            rps.cmd_up(api, up_args(tmp_path, volume_gb=100, volume_mount_path="/workspace"))
        assert api.created_pods == []


class TestTemplateAndRegistry:
    def test_template_carries_the_registry_credential_and_no_pod_volume(self, tmp_path):
        api = FakeAPI()
        rps.cmd_up(api, up_args(tmp_path, volume_gb=0))
        template = api._templates[-1]
        assert template["registry_auth_id"] == "auth1"
        assert template["docker_start_cmd"] == rps.START_CMD
        assert template["volume_in_gb"] == 0
        assert api.created_pods[0]["template_id"] == "tpl1"

    def test_missing_credential_fails_before_anything_is_rented(self, tmp_path):
        api = FakeAPI(creds=[])
        with pytest.raises(SystemExit, match="read:packages"):
            rps.cmd_up(api, up_args(tmp_path, volume_gb=0))
        assert api.created_pods == []

    def test_reuses_a_template_matching_name_and_image(self, tmp_path):
        api = FakeAPI(templates=[{"id": "tpl-old", "name": "distrain-abc1234",
                                  "imageName": "ghcr.io/adamdivak/distrain:abc1234"}])
        rps.cmd_up(api, up_args(tmp_path, volume_gb=0))
        assert api.created_pods[0]["template_id"] == "tpl-old"


class TestGuard:
    def _guard_args(self, tmp_path, **overrides):
        argv = ["--session-file", str(tmp_path / "session.json"), "guard"]
        for key, value in overrides.items():
            argv += ["--" + key.replace("_", "-")] + ([] if value is True else [str(value)])
        args = rps.build_parser().parse_args(argv)
        args.name = "distrain-a100x8"
        return args

    def test_terminates_at_the_ceiling(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture()], ready_pod=pod_fixture())
        past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
        assert rps.cmd_guard(api, self._guard_args(tmp_path, pod_id="pod123",
                                                   deadline=past)) == 2
        assert api.terminated == ["pod123"]

    def test_low_balance_terminates_only_when_asked(self, tmp_path):
        future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        api = FakeAPI(pods=[pod_fixture()], ready_pod=pod_fixture(), balance=5.0)
        assert rps.cmd_guard(api, self._guard_args(
            tmp_path, pod_id="pod123", deadline=future, min_balance=20,
            terminate_on_low_balance=True)) == 3
        assert api.terminated == ["pod123"]

    def test_a_vanished_pod_ends_the_guard(self, tmp_path):
        api = FakeAPI(pods=[], ready_pod=None)
        future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        assert rps.cmd_guard(api, self._guard_args(tmp_path, pod_id="gone",
                                                   deadline=future)) == 0
        assert api.terminated == []

    def test_needs_a_pod_id(self, tmp_path):
        with pytest.raises(SystemExit, match="no pod id"):
            rps.cmd_guard(FakeAPI(), self._guard_args(tmp_path))


class TestDown:
    def _down_args(self, tmp_path, **overrides):
        argv = ["--session-file", str(tmp_path / "session.json"), "down", "-y"]
        for key, value in overrides.items():
            argv += ["--" + key.replace("_", "-")] + ([] if value is True else [str(value)])
        args = rps.build_parser().parse_args(argv)
        args.name = "distrain-a100x8"
        return args

    def test_terminates_the_pod_and_clears_the_session_file(self, tmp_path):
        session = tmp_path / "session.json"
        session.write_text(json.dumps({"pod_id": "pod123"}))
        api = FakeAPI(pods=[pod_fixture()], ready_pod=pod_fixture())
        assert rps.cmd_down(api, self._down_args(tmp_path)) == 0
        assert api.terminated == ["pod123"] and not session.exists()

    def test_volume_survives_teardown_unless_deletion_is_asked_for(self, tmp_path):
        volume = {"id": "vol1", "name": "distrain", "size": 100, "dataCenterId": "US-KS-2"}
        api = FakeAPI(pods=[pod_fixture()], ready_pod=pod_fixture(), volumes=[volume])
        rps.cmd_down(api, self._down_args(tmp_path))
        assert api._volumes == [volume]

        api = FakeAPI(pods=[pod_fixture()], ready_pod=pod_fixture(), volumes=[volume])
        rps.cmd_down(api, self._down_args(tmp_path, delete_volume=True))
        assert api._volumes == []


class TestVerify:
    def _verify_args(self, tmp_path, strict=False, settle=0):
        argv = ["--session-file", str(tmp_path / "session.json"), "verify",
                "--settle-seconds", str(settle)]
        args = rps.build_parser().parse_args(argv + (["--strict"] if strict else []))
        args.name = "distrain-a100x8"
        return args

    def test_an_empty_account_is_clean(self, tmp_path, capsys):
        assert rps.cmd_verify(FakeAPI(), self._verify_args(tmp_path)) == 0
        assert "CLEAN: nothing on this account is costing money." in capsys.readouterr().out

    def test_a_running_pod_fails_with_its_hourly_cost(self, tmp_path, capsys):
        api = FakeAPI(pods=[pod_fixture()])
        assert rps.cmd_verify(api, self._verify_args(tmp_path)) == 1
        out = capsys.readouterr().out
        assert "STILL BILLING" in out and "$12.72/h" in out

    def test_a_stopped_pod_fails_too_because_its_disk_bills(self, tmp_path, capsys):
        api = FakeAPI(pods=[pod_fixture(desiredStatus="EXITED", costPerHr=0)])
        assert rps.cmd_verify(api, self._verify_args(tmp_path)) == 1
        assert "still billing disk" in capsys.readouterr().out

    def test_terminated_pods_are_not_a_leak(self, tmp_path):
        api = FakeAPI(pods=[pod_fixture(desiredStatus="TERMINATED")])
        assert rps.cmd_verify(api, self._verify_args(tmp_path)) == 0

    def test_a_serverless_endpoint_fails(self, tmp_path, capsys):
        api = FakeAPI(endpoints=[{"id": "ep1", "name": "stray"}])
        assert rps.cmd_verify(api, self._verify_args(tmp_path)) == 1
        assert "serverless endpoint(s)" in capsys.readouterr().out

    def test_a_volume_is_reported_and_priced_but_passes(self, tmp_path, capsys):
        volume = {"id": "vol1", "name": "distrain", "size": 100, "dataCenterId": "US-MD-1"}
        assert rps.cmd_verify(FakeAPI(volumes=[volume]), self._verify_args(tmp_path)) == 0
        out = capsys.readouterr().out
        assert "CLEAN by the hour" in out and "$7.00/mo" in out

    def test_strict_makes_the_volume_a_failure(self, tmp_path, capsys):
        volume = {"id": "vol1", "name": "distrain", "size": 100, "dataCenterId": "US-MD-1"}
        api = FakeAPI(volumes=[volume])
        assert rps.cmd_verify(api, self._verify_args(tmp_path, strict=True)) == 1
        assert "--strict" in capsys.readouterr().out

    def test_account_spend_without_a_listed_pod_is_believed(self, tmp_path, capsys):
        """If the money says something is running, the pod list does not get to
        overrule it -- that disagreement is exactly when a check must not pass."""
        api = FakeAPI(reported_spend=3.5)
        assert rps.cmd_verify(api, self._verify_args(tmp_path)) == 1
        assert "no running pod listed" in capsys.readouterr().out

    def test_billing_lag_right_after_a_teardown_is_not_a_leak(self, tmp_path, monkeypatch, capsys):
        """Measured: the account kept reporting the terminated pod's rate for
        ~75 s. Without the settle re-read, every `down` would fail its own check."""
        monkeypatch.setattr(rps.time, "sleep", lambda _s: None)
        api = FakeAPI(reported_spend=[0.50, 0.0])          # stale, then caught up
        assert rps.cmd_verify(api, self._verify_args(tmp_path, settle=45)) == 0
        out = capsys.readouterr().out
        assert "re-reading in 45s" in out and "CLEAN" in out

    def test_spend_that_survives_the_settle_window_is_a_leak(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rps.time, "sleep", lambda _s: None)
        api = FakeAPI(reported_spend=[0.50, 0.50])
        assert rps.cmd_verify(api, self._verify_args(tmp_path, settle=45)) == 1

    def test_down_ends_with_the_check(self, tmp_path, capsys):
        api = FakeAPI(pods=[pod_fixture()], ready_pod=pod_fixture())
        args = rps.build_parser().parse_args(
            ["--session-file", str(tmp_path / "session.json"), "down", "-y"])
        args.name = "distrain-a100x8"
        assert rps.cmd_down(api, args) == 0
        assert "CLEAN" in capsys.readouterr().out


class TestCli:
    def test_pod_name_defaults_to_the_gpu_shape(self, tmp_path, monkeypatch, capsys):
        api = FakeAPI()
        monkeypatch.setattr(rps, "RunpodAPI", lambda: api)
        monkeypatch.setattr(rps, "load_api_key", lambda *a, **k: "test-key")
        exit_code = rps.main(["--session-file", str(tmp_path / "s.json"), "up", "--dry-run",
                              "--volume-gb", "0", "--data-centers", "US-KS-2",
                              "--image", "ghcr.io/adamdivak/distrain:abc1234"])
        assert exit_code == 0
        assert "name: distrain-a100x8" in capsys.readouterr().out

    def test_api_key_comes_from_the_environment_or_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\nRUNPOD_API_KEY='from-dotenv'\n")
        assert rps.load_api_key(env_file) == "from-dotenv"
        monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
        assert rps.load_api_key(env_file) == "from-env"
        monkeypatch.delenv("RUNPOD_API_KEY")
        with pytest.raises(SystemExit):
            rps.load_api_key(tmp_path / "missing.env")
