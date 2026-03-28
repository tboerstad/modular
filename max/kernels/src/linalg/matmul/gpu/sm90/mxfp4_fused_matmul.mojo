# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Fused MXFP4 dequant + FP8 GEMM kernel for SM90 (H100).

Eliminates the global memory roundtrip of the unfused 3-step approach
(dequant MXFP4→FP8, cast BF16→FP8, FP8 GEMM) by dequantizing MXFP4 weights
directly into shared memory within the GEMM producer pipeline.

Architecture:
- Producer warp group (128 threads, CPAsyncBarrierHandler):
  - A tiles: cp.async FP8 activations from global memory (standard path)
  - B tiles: Synchronous MXFP4 dequant → FP8 directly into swizzled smem
- Consumer warp groups: Standard FP8 WGMMA with periodic promotion

This kernel reuses the full HopperMatmulSM90Kernel consumer pipeline.
Only the B-tile loading in the producer is replaced with inline dequant.
"""

from std.math import ceildiv
from std.sys import size_of

from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from std.gpu.globals import WARPGROUP_SIZE
from std.gpu import (
    block_idx_int as block_idx,
    grid_dim_uint as grid_dim,
    thread_idx_int as thread_idx,
)
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import AddressSpace, external_memory
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.host.info import H100
from std.gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from std.gpu.sync import async_copy_arrive
from layout import Layout, LayoutTensor, TileTensor, row_major
from layout.tma_async import create_tma_tile_template, TMATensorTile
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import block_swizzle
from .matmul_kernels import HopperMatmulSM90Kernel, find_K_alignment_upto_16B
from .tile_loader import (
    TileLoaderCPAsync,
    CPAsyncBarrierHandler,
)
from .mxfp4_tile_loader import TileLoaderMXFP4Dequant
from .matmul import _get_c_smem_layout
from structured_kernels.pipeline import ProducerConsumerPipeline
from ....structuring import SMemTile as LTSMemTile


def mxfp4_fused_gemm_sm90[
    c_type: DType,
    a_type: DType,  # FP8 activations (already cast from BF16 by caller)
    //,
    *,
    N: Int,
    K: Int,
    BM: Int = 128,
    BN: Int = 128,
    BK: Int = 128,
    num_pipeline_stages: Int = 4,
    num_consumer: Int = 1,
    wgmma_n: Int = 128,
](
    c_device: TileTensor[c_type, ...],
    a_device: TileTensor[a_type, ...],
    b_packed_device: TileTensor[DType.uint8, ...],
    b_scales_device: TileTensor[DType.float8_e8m0fnu, ...],
    ctx: DeviceContext,
) raises:
    """Fused MXFP4 dequant + FP8 GEMM on SM90.

    Dequantizes MXFP4 weights into shared memory within the GEMM producer,
    eliminating global memory traffic for the dequantized weights.

    Args:
        c_device: Output [M, N] in bfloat16.
        a_device: FP8 activations [M, K].
        b_packed_device: Packed MXFP4 weights [N, K//2] in uint8.
        b_scales_device: E8M0 scales [N, K//SF_VECTOR_SIZE].
        ctx: Device context.
    """
    comptime assert a_type == DType.float8_e4m3fn, "A must be FP8 E4M3"
    comptime assert c_type == DType.bfloat16, "C must be BF16"
    comptime assert BK == 128, "BK must be 128 for FP8"
    comptime assert BN == 128, "BN must be 128 for MXFP4 dequant (1 row/thread)"

    # The GEMM type is FP8×FP8→BF16 (B becomes FP8 after dequant in smem)
    comptime fp8 = DType.float8_e4m3fn

    var a = a_device.to_layout_tensor()
    var c = c_device.to_layout_tensor()
    var b_packed = b_packed_device.to_layout_tensor()
    var b_scales = b_scales_device.to_layout_tensor()

    var M = a.dim[0]()

    comptime block_tile_shape = Index(BM, BN, BK)
    comptime wgmma_shape = Index(64, wgmma_n, 32)
    comptime cluster_shape = StaticTuple[Int32, 3](
        Int32(1), Int32(1), Int32(1)
    )

    comptime c_smem_layout = _get_c_smem_layout[
        block_tile_shape, fp8, fp8, c_type, num_pipeline_stages, 1
    ]()

    # Create the FP8 GEMM kernel type (A=FP8, B=FP8 in smem)
    # We create a "virtual" B layout as if B were [N, K] in FP8, since the
    # consumer sees FP8 tiles in shared memory regardless of the original format.
    comptime b_virtual_layout = Layout.row_major(N, K)

    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    comptime num_threads = WARPGROUP_SIZE * num_consumer + WARPGROUP_SIZE

    comptime KernelType = HopperMatmulSM90Kernel[
        fp8,                    # a_type
        fp8,                    # b_type (FP8 after dequant in smem)
        c_type,
        a.layout,
        b_virtual_layout,       # Virtual FP8 layout for consumer
        c.layout,
        c_smem_layout,
        block_tile_shape,
        wgmma_shape,
        cluster_shape,
        num_pipeline_stages,
        num_threads,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast=False,
        use_tma_store=False,
        promotion_frequency=1,
        pdl_level=PDLLevel(),
    ]

    comptime smem_size = KernelType.SMem.storage_size()
    comptime assert (
        smem_size <= H100.shared_memory_per_multiprocessor - 1024
    ), "SMEM size exceeds H100 limit"

    # C TMA descriptor (for output writes)
    var c_tma_op = create_tma_tile_template[
        c_type,
        2,
        Index(c_smem_layout.shape[0].value(), c_smem_layout.shape[1].value()),
        swizzle_mode=c_swizzle,
        __desc_shape=Index(
            c_smem_layout.shape[0].value(), c_smem_layout.shape[1].value()
        ),
    ]()

    # Launch kernel
    comptime kernel = _mxfp4_fused_kernel[
        KernelType,
        type_of(a).LayoutType,
        type_of(b_packed).LayoutType,
        type_of(b_scales).LayoutType,
        type_of(c_tma_op).rank,
        type_of(c_tma_op).tile_shape,
        type_of(c_tma_op).desc_shape,
    ]

    var grid_dim_val = (ceildiv(N, BN), ceildiv(M, BM))

    ctx.enqueue_function[kernel, kernel](
        c_tma_op,
        a.get_immutable(),
        b_packed.get_immutable(),
        b_scales.get_immutable(),
        c,
        grid_dim=grid_dim_val,
        block_dim=(num_threads,),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(384))
)
def _mxfp4_fused_kernel[
    KernelType: type,
    a_layout: Layout,
    b_packed_layout: Layout,
    b_scales_layout: Layout,
    c_tma_rank: Int,
    c_tile_shape: IndexList[c_tma_rank],
    c_desc_shape: IndexList[c_tma_rank],
](
    c_tma_op: TMATensorTile[
        KernelType.c_type, c_tma_rank, c_tile_shape, c_desc_shape
    ],
    a: LayoutTensor[KernelType.a_type, a_layout, ImmutAnyOrigin],
    b_packed: LayoutTensor[DType.uint8, b_packed_layout, ImmutAnyOrigin],
    b_scales: LayoutTensor[
        DType.float8_e8m0fnu, b_scales_layout, ImmutAnyOrigin
    ],
    c: LayoutTensor[KernelType.c_type, KernelType.c_layout, MutAnyOrigin],
):
    """Fused MXFP4 dequant + FP8 GEMM kernel.

    Producer: cp.async for A (FP8), inline MXFP4→FP8 dequant for B.
    Consumer: Standard FP8 WGMMA with periodic accumulator promotion.
    """
    comptime K = KernelType.b_layout.shape[1].value()
    comptime num_k_iters = ceildiv(K, KernelType.BK)

    # Initialize
    var wgmma_op = KernelType.WgmmaOp()
    ref smem = external_memory[
        Scalar[DType.uint8],
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]().bitcast[KernelType.SMem]()[]

    var (
        warp_group_idx,
        warp_group_thread_idx,
        rank_m,
        rank_n,
        warp_id,
        lane_predicate,
    ) = KernelType.common_kernel_init()

    var pipeline = smem.create_pipeline()
    var barrier_handler = CPAsyncBarrierHandler(
        pipeline, KernelType.num_consumer, KernelType.cluster_size
    )

    # Build A loader (standard cp.async for FP8)
    comptime k_align = find_K_alignment_upto_16B(K * size_of[KernelType.a_type]())
    comptime vector_size = k_align // size_of[KernelType.a_type]()
    comptime num_threads_per_row = KernelType.BK // vector_size
    comptime thread_layout = Layout.row_major(
        WARPGROUP_SIZE // num_threads_per_row, num_threads_per_row
    )
    var a_loader = TileLoaderCPAsync[
        KernelType.a_type,
        a_layout,
        thread_layout,
        KernelType.a_swizzle,
        vector_size,
    ](a)

    # Build B loader (MXFP4 dequant loader)
    var b_loader = TileLoaderMXFP4Dequant[
        b_packed_layout,
        b_scales_layout,
        BN=KernelType.BN,
        BK=KernelType.BK,
    ](b_packed, b_scales)

    KernelType.pipeline_init()

    var block_idx_swizzle = KernelType.get_block_swizzle()

    if warp_group_idx == 0:
        # Producer warp group (all 128 threads participate for cp.async + dequant)
        warpgroup_reg_dealloc[32]()

        KernelType.producer_main_loop_pipeline[num_k_iters=num_k_iters](
            block_idx_swizzle[1],
            block_idx_swizzle[0],
            0,
            a_loader,
            b_loader,
            barrier_handler,
            pipeline,
            smem.a_tiles(),
            smem.b_tiles(),
        )
    else:
        # Consumer warp groups - identical to standard FP8 GEMM
        comptime assert (
            KernelType.num_consumer <= 2
        ), "Only support 1 or 2 consumer"
        warpgroup_reg_alloc[232]()

        var local_warp_group_idx = warp_group_idx - 1
        var c_reg_tile = KernelType.AccumRegTile.stack_allocation()
        var final_c_reg_tile = KernelType.AccumRegTile.stack_allocation()

        KernelType.consumer_arrive_empty_barriers(
            warp_group_thread_idx, pipeline
        )

        KernelType.consumer_main_loop_pipeline[num_k_iters=num_k_iters](
            wgmma_op,
            local_warp_group_idx,
            final_c_reg_tile,
            c_reg_tile,
            pipeline,
            smem.a_tiles(),
            smem.b_tiles(),
            warp_group_thread_idx,
        )

        # FP8 always uses final_c_reg_tile (after promotion)
        var output_reg_tile = final_c_reg_tile

        KernelType.consumer_output(
            c_tma_op,
            c,
            smem.c_tile(),
            output_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            thread_idx.x - WARPGROUP_SIZE,
            block_idx_swizzle[1],
            block_idx_swizzle[0],
        )

    KernelType.finalize_kernel()
