"""The bandwidth convention, checked against a real `nccl-tests` row.

`allreduce_bw.py` exists to produce numbers that sit on the same axis as the
`nccl-tests` output from the 8xA100 anchor. If its convention drifts, a PCIe
point measured with it would be silently incomparable to the NVLink point it is
supposed to be judged against -- so the anchor's own printed row is the fixture.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "allreduce_bw.py"
torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location("allreduce_bw", SCRIPT)
allreduce_bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(allreduce_bw)


def test_ring_factor_matches_the_nccl_tests_convention():
    assert allreduce_bw.ring_factor(8) == pytest.approx(1.75)
    assert allreduce_bw.ring_factor(2) == pytest.approx(1.0)
    assert allreduce_bw.ring_factor(1) == 0.0


def test_busbw_reproduces_the_measured_anchor_row():
    """From out/prime-pcie/session_out/nccl_tests.txt, 8x A100-SXM4-40GB:
    512 MiB in 4806.00 us -> algbw 111.71 GB/s, busbw 195.49 GB/s.
    """
    nbytes, seconds, ranks = 536870912, 4806.00e-6, 8
    assert nbytes / seconds / 1e9 == pytest.approx(111.71, abs=0.01)
    assert allreduce_bw.bus_bandwidth_gbps(nbytes, seconds, ranks) == pytest.approx(
        195.49, abs=0.01
    )


def test_a_two_rank_bus_number_is_directly_comparable_to_an_eight_rank_one():
    """Bus bandwidth is why a 2-GPU box can say anything about an 8-GPU fabric:
    the same wire speed reports the same busbw at either rank count, even though
    the algorithm bandwidth differs by the ring factor.
    """
    nbytes = 512 * 1024**2
    two = allreduce_bw.bus_bandwidth_gbps(nbytes, 1.0, 2)
    eight = allreduce_bw.bus_bandwidth_gbps(nbytes, 1.75, 8)
    assert two == pytest.approx(eight, rel=1e-9)
