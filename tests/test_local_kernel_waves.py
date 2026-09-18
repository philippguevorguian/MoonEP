"""Wave-progress regression for the CTA-independent local kernels.

``dispatch_epilogue`` and ``combine_prologue`` own no grid-wide barrier: every
CTA walks its own round-robin groups and synchronizes only inside its block.
Neither kernel may therefore require whole-grid residency. This test pins that
property by launching both kernels with grids that exceed what the device can
hold at once, including a grid past the shared-memory residency bound, and
requiring exact results within a bounded deadline.

A cooperative launch fails here: CUDA rejects a cooperative grid larger than the
occupancy bound, so the oversized cases error out instead of completing.

Single GPU, no torchrun:

    pytest -s tests/test_local_kernel_waves.py

Set ``MOONEP_WAVE_CASE`` to a printed case index to replay one case. The test
body runs in a spawned subprocess so a hung launch is killed by the deadline
rather than hanging the session.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
import torch.multiprocessing as mp


@dataclass(frozen=True)
class WaveCase:
    kernel: Literal["dispatch_epilogue", "combine_prologue"]
    H: int
    blocks: int
    pdl: bool
    seed: int
    groups: int
    NvS: int
    max_dups: int


def _make_plan(case: WaveCase):
    """Disjoint duplicate groups plus a CPU oracle, with no undefined headers read."""
    from moonep.planning import MoonEPCommPlan

    rng = torch.Generator().manual_seed(case.seed)
    sizes = torch.randint(1, case.max_dups + 1, (case.groups,), generator=rng)
    n_dups = int(sizes.sum())
    slots = torch.randperm(case.NvS, generator=rng)
    primary = slots[: case.groups]
    dups = slots[case.groups : case.groups + n_dups]
    owners = torch.repeat_interleave(torch.arange(case.groups), sizes)

    headers = torch.full((case.NvS, 3), -1, dtype=torch.int32)
    headers[: case.groups, 0] = primary.to(torch.int32)
    headers[: case.groups, 1] = (sizes.cumsum(0) - sizes).to(torch.int32)
    headers[: case.groups, 2] = sizes.to(torch.int32)
    loffs = torch.full((case.NvS,), -1, dtype=torch.int32)
    loffs[:n_dups] = dups.to(torch.int32)

    plan = MoonEPCommPlan(
        dst=torch.zeros(1, dtype=torch.int32, device="cuda"),
        experts_to_copy=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        zero_fill_ranges=torch.zeros((2, 2), dtype=torch.int32, device="cuda"),
        remote_stats=torch.zeros(2, dtype=torch.int32, device="cuda"),
        N=1, R=1, E=1, B=1, NvS=case.NvS, K=case.max_dups + 1,
        dup_groups=headers.cuda(),
        dup_loffs=loffs.cuda(),
        dup_counts=torch.tensor([case.groups, n_dups], dtype=torch.int32, device="cuda"),
    )

    # Small integers keep fp32 accumulation exact on both device and oracle.
    rows = torch.arange(case.NvS, dtype=torch.int32)[:, None]
    cols = torch.arange(case.H, dtype=torch.int32)[None, :]
    source = ((rows * 7 + cols * 3 + case.seed) % 31 - 15).to(torch.bfloat16)
    if case.kernel == "dispatch_epilogue":
        source[dups] = -99  # A skipped copy cannot pass as success.
    expected = source.clone()
    if case.kernel == "dispatch_epilogue":
        expected[dups] = source[primary[owners]]
    else:
        sums = source[primary].float()
        sums.index_add_(0, owners, source[dups].float())
        expected[primary] = sums.to(torch.bfloat16)
    return plan, source, expected


def _worker() -> None:
    from moonep.combine_prologue import CombinePrologueKernel, launch_combine_prologue
    from moonep.dispatch_epilogue import DispatchEpilogueKernel, launch_dispatch_epilogue

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    budget = props.shared_memory_per_block_optin - 1024

    shapes = ((128, 2 * sms + 1), (1024, sms + 1), (2048, 4 * sms + 1), (4096, 2 * sms + 1))
    selected = os.environ.get("MOONEP_WAVE_CASE")
    case_index = 0
    completed = 0

    for kernel in ("dispatch_epilogue", "combine_prologue"):
        cls = DispatchEpilogueKernel if kernel == "dispatch_epilogue" else CombinePrologueKernel
        for H, blocks in shapes:
            B, stages = cls._pick_geometry(H, budget)
            smem = cls._smem_bytes(H, stages, B)
            # Shared memory alone bounds co-residency, whatever the register usage.
            residency_bound = sms * (props.shared_memory_per_multiprocessor // smem)
            if blocks == 4 * sms + 1:
                assert blocks > residency_bound, "wave case must exceed the residency bound"
            max_groups = max(blocks * B + 3, 3 * stages * B + 1)
            max_dups = 3
            NvS = max_groups * (max_dups + 1) + 17
            group_counts = [0, 1, B, max_groups // 2, max_groups]
            for pdl in (False, True):
                for trial, groups in enumerate(group_counts):
                    case = WaveCase(kernel, H, blocks, pdl, 34 + trial, groups, NvS, max_dups)
                    identity = str(case_index)
                    case_index += 1
                    if selected is not None and identity != selected:
                        continue
                    print(f"WAVE_CASE {identity} residency_bound={residency_bound} {case}", flush=True)
                    plan, source, expected = _make_plan(case)
                    hidden = source.cuda()
                    ctx = {"H": H, "R": 1, "NvS": NvS, "num_sms_dedup": blocks,
                           "hidden_buf_local": hidden}
                    if kernel == "dispatch_epilogue":
                        launch_dispatch_epilogue(ctx, plan, pdl_launch=pdl)
                    else:
                        launch_combine_prologue(ctx, plan, pdl_trigger=pdl)
                    done = torch.cuda.current_stream().record_event()
                    deadline = time.monotonic() + 10
                    while not done.query() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    if not done.query():
                        print(f"WAVE_STALL {identity} {case}", flush=True)
                        os._exit(1)
                    torch.testing.assert_close(hidden.cpu(), expected, rtol=0, atol=0)
                    completed += 1

    assert completed > 0, "MOONEP_WAVE_CASE selected no case"
    print(f"WAVE_PASS cases={completed}", flush=True)


def test_local_kernels_progress_in_waves() -> None:
    """Both local kernels complete oversized grids and match the CPU oracle exactly."""
    if "RANK" in os.environ:
        pytest.skip("this test owns its own subprocess")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    pytest.importorskip("moonep")
    proc = mp.get_context("spawn").Process(target=_worker)
    proc.start()
    try:
        proc.join(timeout=600)
        assert not proc.is_alive(), "wave worker exceeded 600 s"
        assert proc.exitcode == 0, f"wave worker failed: exit={proc.exitcode}"
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=5)
