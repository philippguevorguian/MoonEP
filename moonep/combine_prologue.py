"""
MoonEP Combine Prologue — CuTe DSL implementation.

In-place duplicate accumulation on the local NVL shard before combine: for
each duplicate group the primary row and its duplicate rows are streamed
through a smem pipeline, accumulated in fp32 registers, and the partial sum
is stored back to the primary row. The dispatch epilogue fans a primary out
to its duplicates; this kernel is the reverse reduction for the combine
direction.

The caller stages the combine input into the NVL shard first
(``zero_copy=False`` copies the user tensor in with ``tensor.copy_``;
``zero_copy=True`` requires the FFN output to already alias the shard).

This kernel only stages the local rank's combine inputs. Launch it
immediately before ``launch_combine`` on the same stream; the combine kernel
performs the cross-rank publish barrier at entry before consuming peer
ranks' staged rows.
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_ptr

from moonep._common import cp_async_bulk_g2s, pdl_trigger_dependents
from moonep.planning import MoonEPCommPlan


# ============================================================================
# Combine prologue kernel
# ============================================================================

class CombinePrologueKernel:
    """Round-robin fp32 accumulation of duplicate groups on the NVL shard.

    Plan-owned dedup structures drive all work: ``dup_groups`` lists the
    ``(primary_loff, dup_start, dup_n)`` headers, ``dup_loffs`` the compact
    flat duplicate slots, and ``dup_counts[0]`` the valid ``dup_groups``
    prefix length (read on device — no host synchronization).

    Each CTA walks its round-robin groups (group ``i`` -> block
    ``i % num_sms``) as one flat row stream (primary first, then the
    duplicates of each group) and packs ``B`` rows per smem pipeline stage,
    so the mbarrier handshake is amortized over ``B`` rows. Producer and
    consumers count rows with the same ``cnt`` counter, so their stage
    boundaries mirror exactly; a stage may span a group boundary. The
    producer prescans its own group headers once (lane-parallel + butterfly
    reduction) to learn the CTA's total row count, so every stage —
    including the final partial one — announces the exact
    ``expect_tx = min(B, rows_left) * H_BYTES`` before issuing its copies;
    no padding copies and no host-side counts are needed, and the consumers
    simply stop after their last real row.

    Warp layout (5 warps, 160 threads):
      - Warp 0: G2S producer — ``B`` rows per pipeline stage, one
        ``expect_tx`` handshake per stage.
      - Warps 1-4: fp32 ACC consumers — each thread owns
        ``H / ACC_THREADS`` fp32 registers. ``VEC >= 2`` shapes use
        vectorized LDS/STG over ``CHUNKS`` vectors of contiguous elements;
        ``VEC == 1`` falls back to scalar bf16 stores because CuTe DSL
        narrow-precision vector stores require at least 32-bit alignment.

    In-place safety: duplicate rows are only read; a primary row is staged
    strictly before its own consumers write it (the write happens after the
    full-barrier of the stage holding the group's last row, by which time
    every g2s of the group — including the primary — has completed), and no
    other group touches it. The launch grid must equal ``num_sms``.

    This kernel has no cross-rank barrier. Its local pipeline syncs only
    order the in-kernel staging work; ``combine`` publishes the local NVL
    shard with an entry ``cross_rank_barrier`` before reading remote rows.
    """

    ACC_THREADS = 128             # warps 1..4
    num_threads = 32 + ACC_THREADS

    # Handshake amortization vs pipeline depth: prefer the largest per-stage
    # row batch B that still leaves a >= _MIN_STAGES-deep pipeline; fall
    # back to the deepest available geometry when even B=1 cannot reach
    # _MIN_STAGES. (Geometry no longer depends on R — the pre-batching
    # (4,3,2)/(2,) split came from the 05-step study of the un-batched
    # kernel.)
    _B_CANDIDATES = (16, 8, 4, 2, 1)
    _MIN_STAGES = 4

    # LDS/STG vector width (elements): largest of (8, 4, 2, 1) dividing
    # H / ACC_THREADS, i.e. 16/8/4/2-byte accesses.
    _VEC_CANDIDATES = (8, 4, 2, 1)

    def __init__(
        self,
        H: int,
        R: int,
        NvS: int,
        num_sms: int,
        smem_budget: int,
        pdl_trigger: bool,
    ):
        self.H = H
        self.R = R
        self.NvS = NvS
        self.num_sms = num_sms
        self.pdl_trigger = pdl_trigger
        self.B, self.stages_acc = self._pick_geometry(H, smem_budget)
        if self.stages_acc == 0:
            raise RuntimeError(
                f"combine prologue: H={H} too large for per-block smem budget "
                f"{smem_budget} B (need at least {self._smem_bytes(H, 2, 1)} B)"
            )
        assert H % self.ACC_THREADS == 0
        self.VEC = 1
        for v in self._VEC_CANDIDATES:
            if H % (self.ACC_THREADS * v) == 0:
                self.VEC = v
                break

    @staticmethod
    def _smem_bytes(H: int, stages_acc: int, B: int) -> int:
        # acc_smem (bf16, B rows x stages_acc deep) + 2 mbar i64 per stage,
        # padded to 128 B / 16 B, plus 256 B headroom for CUTLASS state.
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a
        return (
            _round_up(stages_acc * B * H * 2, 128)
            + _round_up(stages_acc * 2 * 8, 16)
            + 256
        )

    @classmethod
    def _pick_stages(cls, H: int, smem_budget: int, B: int) -> int:
        for s in (16, 14, 12, 10, 8, 6, 4, 2):
            if cls._smem_bytes(H, s, B) <= smem_budget:
                return s
        return 0

    @classmethod
    def _pick_geometry(cls, H: int, smem_budget: int) -> tuple[int, int]:
        for b in cls._B_CANDIDATES:
            s = cls._pick_stages(H, smem_budget, b)
            if s >= cls._MIN_STAGES:
                return b, s
        return 1, cls._pick_stages(H, smem_budget, 1)

    # ------------------------------------------------------------------ host

    @cute.jit
    def __call__(
        self,
        hidden_buf_local_ptr: cute.Pointer,  # bf16 [NvS, H] local NVL shard
        dup_groups_ptr: cute.Pointer,        # int32 [NvS, 3]
        dup_loffs_ptr: cute.Pointer,         # int32 [NvS,]
        dup_counts_ptr: cute.Pointer,        # int32 [2,]
        stream: cuda.CUstream,
    ):
        H = cutlass.const_expr(self.H)
        NvS = cutlass.const_expr(self.NvS)

        hidden = cute.make_tensor(
            hidden_buf_local_ptr,
            cute.make_layout((NvS, H), stride=(Int64(H), Int64(1))),
        )
        dup_groups_tensor = cute.make_tensor(
            dup_groups_ptr, cute.make_layout((NvS * 3,))
        )
        dup_loffs_tensor = cute.make_tensor(
            dup_loffs_ptr, cute.make_layout((NvS,))
        )
        dup_counts_tensor = cute.make_tensor(
            dup_counts_ptr, cute.make_layout((2,))
        )

        smem_bytes = self._smem_bytes(H, self.stages_acc, self.B)

        self.kernel(
            hidden,
            dup_groups_tensor,
            dup_loffs_tensor,
            dup_counts_tensor,
        ).launch(
            grid=(self.num_sms, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        hidden: cute.Tensor,
        dup_groups_tensor: cute.Tensor,
        dup_loffs_tensor: cute.Tensor,
        dup_counts_tensor: cute.Tensor,
    ):
        H = cutlass.const_expr(self.H)
        H_BYTES = cutlass.const_expr(H * 2)
        stages_acc = cutlass.const_expr(self.stages_acc)
        num_sms = cutlass.const_expr(self.num_sms)
        acc_threads = cutlass.const_expr(self.ACC_THREADS)
        B = cutlass.const_expr(self.B)
        VEC = cutlass.const_expr(self.VEC)
        VEC_BYTES = cutlass.const_expr(VEC * 2)
        CHUNKS = cutlass.const_expr(H // (self.ACC_THREADS * self.VEC))

        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # ----- shared memory layout
        smem = utils.SmemAllocator()

        # Mbarrier block first (16-byte alignment is plenty).
        acc_load_mbar = smem.allocate_array(Int64, num_elems=2 * stages_acc)
        acc_smem = smem.allocate_tensor(
            BFloat16,
            cute.make_ordered_layout((B, H, stages_acc), order=(1, 0, 2)),
            byte_alignment=128,
        )

        # ACC load pipe (TMA G2S → register accumulation):
        #   Producer = warp 0 (1 signalling thread, lane 0).
        #   Consumer = warps 1-4 (128 threads, 4 lane-0 signallers →
        #   consumer_group = 4).  Exact tx = min(B, rows_left) * H_BYTES per
        #   stage (the producer prescans its row count).
        acc_load_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=acc_load_mbar,
            num_stages=stages_acc,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=0,
        )

        n_groups = dup_counts_tensor[0]

        # ============================================
        # Warp 0 — G2S producer: B rows per stage over the flat row stream
        # (each group contributes its primary row first, then its duplicates)
        # ============================================
        if warp_idx == 0:
            acc_load_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages_acc
            )

            lane = cute.arch.lane_idx()

            # Prescan: this CTA's total row count, Σ (dup_count + 1) over
            # its round-robin groups. Lane l sums groups
            # bidx + (32*j + l) * num_sms; the butterfly reduction leaves
            # the warp-uniform total in every lane. Headers are 12 B each
            # and re-read by the main loop right after — cache-hot, so the
            # prescan costs ~n_groups / (num_sms * 32) iterations.
            rows_left = Int32(0)
            gj = Int32(bidx + lane * num_sms)
            while gj < n_groups:
                rows_left += dup_groups_tensor[gj * 3 + 2] + Int32(1)
                gj += Int32(num_sms * 32)
            for off in (16, 8, 4, 2, 1):
                rows_left += cute.arch.shuffle_sync_bfly(rows_left, Int32(off))

            gi = Int32(bidx)
            lane = cute.arch.lane_idx()
            cnt = Int32(0)

            pref_dup_groups_base = gi * 3
            pref_dup_group_val = Int32(0)
            # gi may already be past the compact prefix (empty CTA): the
            # header row would be garbage and its dup_start would feed the
            # dependent dup_loffs load below — guard the load so row_pref
            # derives from an all-zero header instead.
            if gi < n_groups:
                if lane < Int32(3):
                    pref_dup_group_val = dup_groups_tensor[pref_dup_groups_base + lane]
            primary_loff = cute.arch.shuffle_sync(pref_dup_group_val, Int32(0))
            dup_start_loff = cute.arch.shuffle_sync(pref_dup_group_val, Int32(1))
            dup_count = cute.arch.shuffle_sync(pref_dup_group_val, Int32(2))
            row_pref = primary_loff
            if lane > Int32(0) and lane <= dup_count:
                row_pref = dup_loffs_tensor[dup_start_loff + lane - 1]

            while gi < n_groups:
                gi_next = gi + Int32(num_sms)
                next_dup_group_base = gi_next * 3
                next_dup_group_val = Int32(0)
                # Last group of this CTA: gi_next is past the compact prefix,
                # so the header row is garbage and the loop-tail dup_loffs
                # load would index with a garbage dup_start — guard the load
                # (an all-zero header makes the tail recompute read nothing).
                if gi_next < n_groups:
                    if lane < Int32(3):
                        next_dup_group_val = dup_groups_tensor[next_dup_group_base + lane]

                for k in cutlass.range(dup_count + 1, unroll=1):
                    # Lane j holds row j, so the shuffle covers k <= 31; the
                    # launcher asserts plan.K <= 32 (dup_count <= 31).
                    row_loff = cute.arch.shuffle_sync(row_pref, k)

                    if cnt == Int32(0):
                        acc_load_pipe.sync_object_empty.wait(
                            acc_load_state.index, acc_load_state.phase
                        )
                        if lane == 0:
                            # rows_left still includes the current row, so
                            # the final partial stage announces exactly its
                            # real row count — the full barrier always
                            # completes without padding copies.
                            tx_rows = Int32(B)
                            if rows_left < tx_rows:
                                tx_rows = rows_left
                            mbar = acc_load_pipe.producer_get_barrier(
                                acc_load_state
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar, tx_rows * H_BYTES
                            )

                    if lane == 0:
                        g_row_int = (
                            hidden.iterator + Int64(row_loff) * Int64(H)
                        ).toint()
                        s_row_int = (
                            acc_smem.iterator
                            + acc_load_state.index * B * H + cnt * H
                        ).toint()
                        mbar_int = acc_load_pipe.producer_get_barrier(
                            acc_load_state
                        ).toint()
                        cp_async_bulk_g2s(
                            s_row_int.ir_value(),
                            g_row_int.ir_value(),
                            Int32(H_BYTES).ir_value(),
                            mbar_int.ir_value(),
                        )

                    rows_left -= Int32(1)
                    cnt += Int32(1)
                    if cnt == Int32(B):
                        cnt = Int32(0)
                        acc_load_state.advance()

                primary_loff = cute.arch.shuffle_sync(next_dup_group_val, Int32(0))
                dup_start_loff = cute.arch.shuffle_sync(next_dup_group_val, Int32(1))
                dup_count = cute.arch.shuffle_sync(next_dup_group_val, Int32(2))
                row_pref = primary_loff
                if lane > Int32(0) and lane <= dup_count:
                    row_pref = dup_loffs_tensor[dup_start_loff + lane - 1]
                gi += Int32(num_sms)

        # ============================================
        # Warps 1...4 — fp32 ACC consumers (accumulate in registers)
        # ============================================
        elif warp_idx >= 1 and warp_idx <= 4:
            acc_use_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages_acc
            )

            acc_tid = tidx - 32  # 0..127

            # VEC==1 cannot use Tensor.store of a single bf16 element: CuTe
            # requires narrow-precision vector stores to be 32-bit aligned.
            # Keep that shape on the original scalar path; VEC>=2 uses the
            # vectorized path below.
            if cutlass.const_expr(VEC == 1):
                acc_reg = cute.make_rmem_tensor(
                    cute.make_layout((CHUNKS,)), Float32
                )
            else:
                acc_reg = cute.make_rmem_tensor(
                    cute.make_layout((VEC, CHUNKS)), Float32
                )

            gi = Int32(bidx)
            cnt = Int32(0)
            lane = cute.arch.lane_idx()

            pref_dup_group_base = gi * 3
            pref_dup_group_val = Int32(0)
            if gi < n_groups:
                if lane < Int32(3):
                    pref_dup_group_val = dup_groups_tensor[pref_dup_group_base + lane]
            primary_loff = cute.arch.shuffle_sync(pref_dup_group_val, Int32(0))
            dup_count = cute.arch.shuffle_sync(pref_dup_group_val, Int32(2))

            while gi < n_groups:
                next_gi = gi + Int32(num_sms)
                next_dup_group_base = next_gi * 3
                next_dup_group_val = Int32(0)
                if next_gi < n_groups:
                    if lane < Int32(3):
                        next_dup_group_val = dup_groups_tensor[next_dup_group_base + lane]

                # Prologue: zero accumulator regs.
                # Thread-local — no cross-warp barrier needed.
                acc_reg.fill(0.0)

                for k in cutlass.range(dup_count + 1, unroll=1):
                    if cnt == Int32(0):
                        acc_load_pipe.consumer_wait(acc_use_state)

                    row_ptr = (
                        acc_smem.iterator
                        + acc_use_state.index * B * H + cnt * H
                    )
                    if cutlass.const_expr(VEC == 1):
                        for c in cutlass.range_constexpr(CHUNKS):
                            idx = c * acc_threads + acc_tid
                            acc_reg[c] = acc_reg[c] + Float32(
                                acc_smem[cnt, idx, acc_use_state.index]
                            )
                    else:
                        # fp32 accumulate: acc_reg[:, c] += f32(stage row
                        # chunk)
                        for c in cutlass.range_constexpr(CHUNKS):
                            s_vec = cute.make_tensor(
                                cute.make_ptr(
                                    BFloat16,
                                    (
                                        row_ptr
                                        + c * acc_threads * VEC + acc_tid * VEC
                                    ).toint(),
                                    cute.AddressSpace.smem,
                                    assumed_align=VEC_BYTES,
                                ),
                                cute.make_layout((VEC,)),
                            )
                            acc_slice = acc_reg[(None, c)]
                            acc_slice.store(
                                acc_slice.load() + s_vec.load().to(Float32)
                            )

                    cnt += Int32(1)
                    if cnt == Int32(B):
                        cnt = Int32(0)
                        # ACC warps must finish reading the stage before
                        # consumer_release lets the producer overwrite it.
                        # consumer_release signals from each warp's lane 0
                        # only (PipelineTmaAsync signalling thread;
                        # consumer_group=4), so a warp-level sync makes that
                        # arrive cover all 32 lanes' reads — cross-warp
                        # completion is already gated by the 4-count empty
                        # mbarrier.
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_warp()
                        acc_load_pipe.consumer_release(acc_use_state)
                        acc_use_state.advance()

                # Epilogue: fp32 → bf16 back to the primary row of the NVL
                # shard. Use scalar stores for VEC==1; vectorized stores for
                # VEC>=2.
                out_row_ptr = hidden.iterator + Int64(primary_loff) * Int64(H)
                if cutlass.const_expr(VEC == 1):
                    for c in cutlass.range_constexpr(CHUNKS):
                        idx = c * acc_threads + acc_tid
                        hidden[primary_loff, idx] = BFloat16(acc_reg[c])
                else:
                    for c in cutlass.range_constexpr(CHUNKS):
                        g_vec = cute.make_tensor(
                            cute.make_ptr(
                                BFloat16,
                                (
                                    out_row_ptr
                                    + c * acc_threads * VEC + acc_tid * VEC
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=VEC_BYTES,
                            ),
                            cute.make_layout((VEC,)),
                        )
                        g_vec.store(acc_reg[(None, c)].load().to(BFloat16))

                primary_loff = cute.arch.shuffle_sync(next_dup_group_val, Int32(0))
                dup_count = cute.arch.shuffle_sync(next_dup_group_val, Int32(2))
                gi += Int32(num_sms)

        if cutlass.const_expr(self.pdl_trigger):
            pdl_trigger_dependents(tidx)


# ============================================================================
# Host launcher with per-shape compile cache
# ============================================================================

@functools.lru_cache(maxsize=None)
def _max_smem_per_block_optin(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


def _check_prologue_plan(ctx: dict, plan: MoonEPCommPlan, dev: torch.device) -> None:
    assert isinstance(plan, MoonEPCommPlan), "plan must be a MoonEPCommPlan"
    NvS = int(ctx['NvS'])

    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"

    def _check_tensor(t: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"
        assert tuple(t.shape) == shape, \
            f"{name} must be shape {shape}, got {tuple(t.shape)}"
        assert t.device == dev, \
            f"{name} must be on {dev}, got {t.device}"

    _check_tensor(plan.dup_groups, "dup_groups", (NvS, 3))
    _check_tensor(plan.dup_loffs, "dup_loffs", (NvS,))
    _check_tensor(plan.dup_counts, "dup_counts", (2,))


@functools.lru_cache(maxsize=None)
def _get_compiled(
    H: int,
    R: int,
    NvS: int,
    num_sms: int,
    device_index: int,
    pdl_trigger: bool,
):
    smem_budget = _max_smem_per_block_optin(device_index) - 1024
    kernel = CombinePrologueKernel(
        H=H, R=R, NvS=NvS, num_sms=num_sms, smem_budget=smem_budget,
        pdl_trigger=pdl_trigger,
    )

    bf16_ptr = make_ptr(BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    stream_arg = cuda.CUstream(0)

    return cute.compile(
        kernel,
        bf16_ptr,  # hidden_buf_local
        i32_ptr,   # dup_groups
        i32_ptr,   # dup_loffs
        i32_ptr,   # dup_counts
        stream_arg,
    )


def launch_combine_prologue(
    ctx: dict,
    plan: MoonEPCommPlan,
    *,
    pdl_trigger: bool = False,
):
    """Accumulate duplicate rows into their primary row on the NVL shard.

    Must run after the combine input has been staged into
    ``ctx['hidden_buf_local']`` and before ``launch_combine`` on the same
    stream. Consumes the plan-owned ``dup_groups`` / ``dup_loffs`` /
    ``dup_counts`` from a fresh dispatch builder (reuse paths pass the saved
    tensors unchanged).
    """
    H = int(ctx['H'])
    R = int(ctx['R'])
    NvS = int(ctx['NvS'])
    num_sms = int(ctx['num_sms_dedup'])
    dev = ctx['hidden_buf_local'].device
    device_index = dev.index

    assert H % CombinePrologueKernel.ACC_THREADS == 0, \
        f"H must be a multiple of ACC_THREADS={CombinePrologueKernel.ACC_THREADS} " \
        "(also covers the 16-B bulk-copy alignment requirement)"
    assert ctx['hidden_buf_local'].dtype == torch.bfloat16 and \
        ctx['hidden_buf_local'].is_contiguous()
    assert ctx['hidden_buf_local'].is_cuda, "hidden_buf_local must be a CUDA tensor"
    assert tuple(ctx['hidden_buf_local'].shape) == (NvS, H), \
        f"hidden_buf_local must be shape ({NvS}, {H}), " \
        f"got {tuple(ctx['hidden_buf_local'].shape)}"
    assert device_index is not None, "combine prologue requires a CUDA device"
    assert plan.K <= 32, \
        "combine prologue's lane-owned row-loff prefetch covers dup_count <= 31 " \
        f"(shuffle width); got K={plan.K}"
    _check_prologue_plan(ctx, plan, dev)

    prologue_compiled = _get_compiled(H, R, NvS, num_sms, device_index, bool(pdl_trigger))

    hbf_local_ptr = make_ptr(BFloat16, ctx['hidden_buf_local'].data_ptr(),
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_groups_ptr = make_ptr(Int32, plan.dup_groups.data_ptr(),
                              cute.AddressSpace.gmem, assumed_align=16)
    dup_loffs_ptr = make_ptr(Int32, plan.dup_loffs.data_ptr(),
                             cute.AddressSpace.gmem, assumed_align=16)
    dup_counts_ptr = make_ptr(Int32, plan.dup_counts.data_ptr(),
                              cute.AddressSpace.gmem, assumed_align=8)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    prologue_compiled(
        hbf_local_ptr,
        dup_groups_ptr,
        dup_loffs_ptr,
        dup_counts_ptr,
        stream,
    )
