"""DiLoCo correctness tests (decisions.md section 16 test matrix), gloo/CPU.

Two tiers, mirroring test_ddp.py: direct synchronizer tests in a world-size-1
process group (the outer step's math against hand-computed Nesterov updates),
and spawned 2-rank runs through train() for the cross-rank invariants.

The world-size-1 tier is legitimate because an all-reduce over one rank is the
identity: everything specific to DiLoCo -- delta computation, the outer
optimizer, round arithmetic, the copy-back -- runs exactly as it does at any K,
while the assertions stay in the test process instead of behind spawn.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import distributed as dist

from distrain.ddp_synchronizer import DiLoCoSynchronizer
from distrain.model import GPT, GPTConfig
from distrain.train import (
    TrainConfig,
    cleanup_distributed,
    resolve_device,
    resolve_distributed_synchronizer,
    setup_distributed,
    train,
)
from helpers import tiny_train_config


def _tiny_gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=16, vocab_size=32, n_layer=1, n_head=2,
                         n_embd=16, dropout=0.0, bias=False))


def _params(model) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _perturb(model, seed: int, scale: float = 0.01) -> list[torch.Tensor]:
    """Deterministically shift every parameter in place; returns the shifts.

    Stands in for "H inner steps moved the replica": the outer step only sees
    the displacement, so how it was produced is irrelevant to its math.
    """
    torch.manual_seed(seed)
    shifts = []
    with torch.no_grad():
        for p in model.parameters():
            d = scale * torch.randn_like(p)
            p.add_(d)
            shifts.append(d)
    return shifts


@pytest.fixture
def single_rank_group(tmp_path):
    setup_distributed(1, 0, "cpu", init_method=f"file://{tmp_path}/rdzv_ws1")
    yield
    cleanup_distributed()


class TestConfigValidation:
    """diloco's startup guards (decisions.md section 16): fail loudly, before any rented clock."""

    def test_val_cadence_must_be_multiple_of_sync_interval(self, tiny_data):
        cfg = tiny_train_config(tiny_data, world_size=2, distributed_mode="diloco",
                                val_every=15, outer_sync_every=10)
        with pytest.raises(ValueError, match="[Vv]alidation"):
            resolve_distributed_synchronizer(cfg, model=None)

    @pytest.mark.parametrize("missing", ["outer_sync_every", "outer_lr", "outer_moment"])
    def test_outer_hyperparameters_must_be_set(self, tiny_data, missing):
        cfg = tiny_train_config(tiny_data, world_size=2, distributed_mode="diloco",
                                val_every=0, **{missing: None})
        with pytest.raises(ValueError, match=missing):
            resolve_distributed_synchronizer(cfg, model=None)

    def test_unknown_mode_is_rejected(self, tiny_data):
        cfg = tiny_train_config(tiny_data, world_size=2, distributed_mode="ddp_typo")
        with pytest.raises(ValueError, match="ddp_typo"):
            resolve_distributed_synchronizer(cfg, model=None)


class TestOuterStepMath:
    """Section 16 tests 1 and 4, at world size 1.

    Comparisons use assert_close with tight tolerances rather than bitwise
    equality: the outer step round-trips through the delta
    (theta_new = theta0 - lr * (theta0 - theta_inner)), and a - (a - b) == b
    does not hold bitwise in floating point even at lr=1.
    """

    def test_lr1_momentum0_is_parameter_averaging(self, single_rank_group):
        # With outer lr 1.0 and momentum 0 the update is theta0 - mean(delta),
        # i.e. plain parameter averaging; at K=1 that is the inner result itself,
        # so the sync must be (numerically) a no-op on the trajectory.
        model = _tiny_gpt()
        sync = DiLoCoSynchronizer(model, world_size=1, outer_sync_every=4,
                                  outer_lr=1.0, outer_moment=0.0)
        _perturb(model, seed=1)
        inner = _params(model)

        sync.maybe_outer_step(step=4, is_last=False)  # step % 4 == 0, step > 0: boundary

        for got, want in zip(_params(model), inner):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)
        # the baseline for the next round is the freshly synced state
        for copy, live in zip(sync.model_copy, model.parameters()):
            assert torch.equal(copy, live.detach())

    def test_no_sync_off_boundary(self, single_rank_group):
        # Section 16 test 4: the outer step fires only at positive multiples of
        # H. Step 0 is deliberately in the no-fire list: no round has finished
        # there, and the step-0 val is a per-replica fingerprint, not a boundary.
        # Off-boundary calls must leave both the live model and the baseline
        # bit-identical -- "did not touch" is exact, unlike the update math.
        model = _tiny_gpt()
        sync = DiLoCoSynchronizer(model, world_size=1, outer_sync_every=4,
                                  outer_lr=0.7, outer_moment=0.9)
        baseline = [c.clone() for c in sync.model_copy]
        _perturb(model, seed=1)
        perturbed = _params(model)

        for step in (0, 1, 2, 3, 5):  # step 0, or step % 4 != 0
            sync.maybe_outer_step(step, is_last=False)
            for got, want in zip(_params(model), perturbed):
                assert torch.equal(got, want), f"outer step fired at step {step}"
            for copy, want in zip(sync.model_copy, baseline):
                assert torch.equal(copy, want)

    def test_is_last_forces_sync(self, single_rank_group):
        # The final iteration always syncs, boundary or not, so the final val
        # (and the reported crossing) is measured on the shared model.
        model = _tiny_gpt()
        sync = DiLoCoSynchronizer(model, world_size=1, outer_sync_every=1000,
                                  outer_lr=1.0, outer_moment=0.0)
        theta0 = [c.clone() for c in sync.model_copy]
        _perturb(model, seed=1)

        sync.maybe_outer_step(step=2, is_last=True)  # far from any boundary

        moved = any(not torch.equal(copy, t0)
                    for copy, t0 in zip(sync.model_copy, theta0))
        assert moved, "is_last=True did not trigger the outer step"
        for copy, live in zip(sync.model_copy, model.parameters()):
            assert torch.equal(copy, live.detach())

    def test_nesterov_momentum_across_rounds(self, single_rank_group):
        # Pins the outer update to hand-computed Nesterov SGD, two rounds deep,
        # so the momentum buffer's persistence across rounds is what's tested.
        # PyTorch's SGD(nesterov=True): buf <- mu*buf + g; update = g + mu*buf.
        # Round 1 (empty buffer), with g1 = theta0 - (theta0 + d1) = -d1:
        #   buf = -d1; update = -(1 + mu)*d1; theta1 = theta0 + lr*(1 + mu)*d1
        # Round 2, with g2 = -d2:
        #   buf = -mu*d1 - d2; update = -(mu^2*d1 + (1 + mu)*d2)
        #   theta2 = theta1 + lr*(mu^2*d1 + (1 + mu)*d2)
        lr, mu = 0.7, 0.9
        model = _tiny_gpt()
        sync = DiLoCoSynchronizer(model, world_size=1, outer_sync_every=1,
                                  outer_lr=lr, outer_moment=mu)
        theta0 = [c.clone() for c in sync.model_copy]

        d1 = _perturb(model, seed=1)
        sync.maybe_outer_step(step=1, is_last=False)
        theta1 = _params(model)
        for got, t0, a in zip(theta1, theta0, d1):
            torch.testing.assert_close(got, t0 + lr * (1 + mu) * a,
                                       rtol=1e-5, atol=1e-7)

        d2 = _perturb(model, seed=2)
        sync.maybe_outer_step(step=2, is_last=False)
        for got, t1, a, b in zip(_params(model), theta1, d1, d2):
            torch.testing.assert_close(got, t1 + lr * (mu**2 * a + (1 + mu) * b),
                                       rtol=1e-5, atol=1e-7)


# --- spawned 2-rank tests -----------------------------------------------------

def _train_and_save_params(rank, cfg: TrainConfig, tmp_path, tag,
                           force_rank0_data, fault_rank):
    true_rank = rank
    try:
        cfg.rank = rank
        setup_distributed(cfg.world_size, cfg.rank,
                          resolve_device(cfg.device, cfg.local_rank),
                          init_method=f"file://{tmp_path}/rdzv_{tag}")

        if fault_rank is not None and true_rank == fault_rank:
            # Corrupt this rank's view of the averaged delta *after* the real
            # collective: it then applies a different outer update than its
            # peers, which the replica-equality assertion must catch. Exists to
            # prove that assertion is not vacuous (section 10's mutation rule).
            real_all_reduce = dist.all_reduce

            def corrupting_all_reduce(tensor, *args, **kwargs):
                work = real_all_reduce(tensor, *args, **kwargs)
                tensor.add_(1e-3)
                return work

            dist.all_reduce = corrupting_all_reduce

        if force_rank0_data:
            # Every replica consumes rank 0's data stream (and seed): replicas
            # then compute identical deltas, making the outer average degenerate.
            # The process group still uses the true rank; only train() is fooled.
            cfg.rank = 0

        results = train(cfg, return_debug_values=["model"])
        torch.save(OrderedDict(results["model"].named_parameters()),
                   tmp_path / f"params_{tag}_rank{true_rank}.pt")
    finally:
        cleanup_distributed()


def _spawn_train(cfg, tmp_path, tag, force_rank0_data=False, fault_rank=None):
    torch.multiprocessing.spawn(
        _train_and_save_params,
        (cfg, tmp_path, tag, force_rank0_data, fault_rank),
        nprocs=cfg.world_size)


def _load_params(tmp_path, tag, rank) -> OrderedDict:
    return torch.load(tmp_path / f"params_{tag}_rank{rank}.pt", weights_only=True)


def _diloco_cfg(tiny_data, **overrides):
    cfg = dict(world_size=2, distributed_mode="diloco", outer_sync_every=2,
               max_steps=4, val_every=2)
    cfg.update(overrides)
    return tiny_train_config(tiny_data, **cfg)


class TestReplicaEquality:
    """Section 16 test 3: parameters bitwise equal across ranks at boundaries.

    The all-reduce hands every rank the identical averaged delta and the outer
    SGD is deterministic, so equality after the final sync is exact, not
    approximate. Inner optimizer state is *expected* to differ across ranks and
    is deliberately never compared (section 16, load-bearing).
    """

    @pytest.mark.parametrize("grad_accum_steps", [1, 2])
    def test_replicas_equal_after_final_sync(self, tiny_data, tmp_path, grad_accum_steps):
        cfg = _diloco_cfg(tiny_data, grad_accum_steps=grad_accum_steps)
        _spawn_train(cfg, tmp_path, tag=f"eq{grad_accum_steps}")

        p0 = _load_params(tmp_path, f"eq{grad_accum_steps}", 0)
        p1 = _load_params(tmp_path, f"eq{grad_accum_steps}", 1)
        for key in p0:
            assert torch.equal(p0[key], p1[key]), f"replicas diverged at {key}"

    def test_equality_check_has_teeth(self, tiny_data, tmp_path):
        cfg = _diloco_cfg(tiny_data)
        _spawn_train(cfg, tmp_path, tag="fault", fault_rank=1)

        p0 = _load_params(tmp_path, "fault", 0)
        p1 = _load_params(tmp_path, "fault", 1)
        assert any(not torch.equal(p0[key], p1[key]) for key in p0), (
            "corrupting rank 1's averaged delta did not break replica "
            "equality -- the equality test is vacuous")


class TestDegenerateAveraging:
    """Section 16 test 2: identical data on every replica collapses DiLoCo.

    With both replicas fed rank 0's stream, each computes the same delta, the
    average equals the local delta, and an outer step at lr=1/momentum=0 just
    installs the inner result -- the same trajectory ddp_naive produces, since
    averaging two identical gradients is also the identity. The two modes must
    then agree up to the outer step's floating-point round trip.
    """

    def test_matches_ddp_on_identical_data(self, tiny_data, tmp_path):
        common = dict(world_size=2, max_steps=4, val_every=2)
        ddp = tiny_train_config(tiny_data, distributed_mode="ddp_naive", **common)
        diloco = tiny_train_config(tiny_data, distributed_mode="diloco",
                                   outer_sync_every=2, outer_lr=1.0,
                                   outer_moment=0.0, **common)
        _spawn_train(ddp, tmp_path, tag="degen_ddp", force_rank0_data=True)
        _spawn_train(diloco, tmp_path, tag="degen_diloco", force_rank0_data=True)

        p_ddp = _load_params(tmp_path, "degen_ddp", 0)
        p_diloco = _load_params(tmp_path, "degen_diloco", 0)
        for key in p_ddp:
            torch.testing.assert_close(
                p_diloco[key], p_ddp[key], rtol=1e-5, atol=1e-6,
                msg=lambda m, key=key: f"{key}: {m}")


def _eval_placement_worker(rank, cfg: TrainConfig, tmp_path, tag):
    try:
        cfg.rank = rank
        setup_distributed(cfg.world_size, cfg.rank,
                          resolve_device(cfg.device, cfg.local_rank),
                          init_method=f"file://{tmp_path}/rdzv_{tag}")

        # Instrument train() in this spawned process: capture the synchronizer
        # at construction, and record at every evaluate() call whether the live
        # model equals the synced baseline (true exactly when the last event
        # touching the params was an outer step, not an inner step).
        import distrain.train as train_module

        holder = {}
        real_resolve = train_module.resolve_distributed_synchronizer

        def capturing_resolve(cfg_, model):
            holder["sync"] = real_resolve(cfg_, model)
            return holder["sync"]

        train_module.resolve_distributed_synchronizer = capturing_resolve

        synced_at_eval = []
        real_evaluate = train_module.evaluate

        def recording_evaluate(*args, **kwargs):
            sync = holder["sync"]
            synced_at_eval.append(all(
                torch.equal(live.detach(), copy)
                for live, copy in zip(sync.model.parameters(), sync.model_copy)))
            return real_evaluate(*args, **kwargs)

        train_module.evaluate = recording_evaluate

        train(cfg)
        if rank == 0:  # only the primary evaluates; other ranks record nothing
            torch.save(synced_at_eval, tmp_path / f"eval_synced_{tag}.pt")
    finally:
        cleanup_distributed()


class TestEvalPlacement:
    """Section 16 test 6: every reported val point is the shared model.

    Mid-round, rank 0's replica has diverged from the synced state, and the
    <= 3.28 crossing check runs at every eval -- an eval that sees a diverged
    replica can cross before the shared model does and silently corrupt the
    headline metric. So under diloco, every evaluate() call must land on
    parameters identical to the synced baseline.

    One deliberate carve-out: the step-0 val. No round has finished there, so
    there is no synced model to show it; it is rank 0's replica after one inner
    step -- the same one-step init fingerprint a DDP run prints, kept because
    the eval schedule is mode-independent by design. It cannot record a real
    crossing (val is ~10.8 at step 0).
    """

    def test_boundary_evals_see_synced_params(self, tiny_data, tmp_path):
        cfg = _diloco_cfg(tiny_data)
        torch.multiprocessing.spawn(_eval_placement_worker,
                                    (cfg, tmp_path, "evalsync"),
                                    nprocs=cfg.world_size)

        synced_at_eval = torch.load(tmp_path / "eval_synced_evalsync.pt",
                                    weights_only=True)
        # with val_every=2, max_steps=4: evals at steps 0, 2 and 3 (is_last)
        assert len(synced_at_eval) >= 3, "expected three evals in the run"
        assert all(synced_at_eval[1:]), (
            f"evals on synced model: {synced_at_eval} -- an evaluation past "
            f"step 0 ran on a mid-round replica instead of the synced model; "
            f"the outer step must fire in the same iteration as the val "
            f"(step % H == 0), before it.")


class TestResume:
    """Section 16 test 5: resume must be exact on every rank.

    diloco checkpoints are per-rank files (each rank's replica + inner AdamW
    state, which genuinely differ across ranks) with the outer state -- the
    round-start baseline and the outer momentum -- in rank 0's file only,
    broadcast on restore. --resume-from names any rank's file; each rank
    substitutes its own rank suffix.

    The mid-round case is the sharp one: there the checkpointed replica params
    differ from the round-start baseline, so a resume that re-derives the
    baseline from the loaded params (instead of restoring it) shrinks the
    round's delta to only the post-resume steps and silently changes the
    trajectory -- params alone would not catch a missing momentum restore
    either, hence the bitwise comparison against the uninterrupted run.
    """

    @pytest.mark.parametrize("sync_every,resume_step",
                             [(2, 3), (4, 6)], ids=["boundary", "mid-round"])
    def test_resume_is_bit_exact_per_rank(self, tiny_data, tmp_path,
                                          sync_every, resume_step):
        # syncs fire at the end of iterations H, 2H, ...; a checkpoint at
        # next_step=N holds the state after iteration N-1, so N=3 with H=2 is a
        # just-synced boundary save while N=6 with H=4 is mid-round (the sync
        # fired at iteration 4, iteration 5 has diverged again)
        ckpt_dir = tmp_path / f"ckpt_h{sync_every}"
        common = dict(outer_sync_every=sync_every, val_every=0, max_steps=8,
                      checkpoint_dir=str(ckpt_dir), run_name="diloco-resume",
                      checkpoint_keep_last=10)
        full = _diloco_cfg(tiny_data, checkpoint_every=3, **common)
        _spawn_train(full, tmp_path, tag=f"resume_full_h{sync_every}")

        resume_file = ckpt_dir / f"diloco-resume-step{resume_step:06d}-rank0.pt"
        assert resume_file.exists(), "expected a per-rank checkpoint from rank 0"
        assert (ckpt_dir / f"diloco-resume-step{resume_step:06d}-rank1.pt").exists()

        resumed = _diloco_cfg(tiny_data, checkpoint_every=0,
                              resume_from=str(resume_file), **common)
        _spawn_train(resumed, tmp_path, tag=f"resume_cont_h{sync_every}")

        for rank in (0, 1):
            p_full = _load_params(tmp_path, f"resume_full_h{sync_every}", rank)
            p_resumed = _load_params(tmp_path, f"resume_cont_h{sync_every}", rank)
            for key in p_full:
                assert torch.equal(p_full[key], p_resumed[key]), (
                    f"rank {rank} diverged after resume at {key}")
