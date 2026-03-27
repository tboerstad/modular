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
"""MXFP4 grouped matmul on H100 (SM90) via dequant-to-FP8 + FP8 grouped GEMM.

Dequantizes MXFP4 weights to FP8 for all experts, casts BF16 activations to
FP8, then dispatches the SM90 warp-specialized FP8 grouped GEMM.
"""

from std.algorithm.functional import elementwise
from std.collections import Optional
from std.gpu.host import DeviceContext
from std.sys.info import _accelerator_arch, simd_width_of
from layout import Coord, Idx, TileTensor, row_major
from std.utils.index import Index, IndexList

from .mxfp4_dequant import dequant_mxfp4
from .matmul.gpu.sm90.grouped_matmul import grouped_matmul_sm90
from .matmul.gpu.sm90.dispatch import _find_largest_bn_for_sm90_matmul
from .utils import elementwise_epilogue_type


def mxfp4_grouped_matmul_sm90[
    *,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b_packed: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    expert_ids: TileTensor[
        mut=False, DType.int32, address_space=AddressSpace.GENERIC, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """MXFP4 grouped matmul: dequant B weights to FP8, cast A to FP8, SM90
    FP8 grouped GEMM.

    Args:
        c: Output [total_tokens, N] in bfloat16.
        a: Activations [total_tokens, K] in bfloat16.
        b_packed: Weights [num_experts, N, K//2] in uint8 (packed MXFP4).
        b_scales: Weight scales [num_experts, N, K//32] in float8_e8m0fnu.
        a_offsets: Token offsets per expert [num_active_experts + 1] in uint32.
        expert_ids: Expert indices [num_active_experts] in int32.
        max_num_tokens_per_expert: Maximum tokens assigned to any single expert.
        num_active_experts: Number of active experts in this batch.
        ctx: Device context.
    """
    comptime assert (
        "sm_90" in _accelerator_arch()
    ), "mxfp4_grouped_matmul_sm90 requires SM90"

    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b_packed.dtype
    comptime b_scales_type = b_scales.dtype

    comptime assert c_type == DType.bfloat16, "output must be bfloat16"
    comptime assert a_type == DType.bfloat16, "activations must be bfloat16"
    comptime assert b_type == DType.uint8, "weights must be uint8 (packed FP4)"
    comptime assert (
        b_scales_type == DType.float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"

    # Early exit for empty inputs.
    if num_active_experts == 0 or Int(a.dim[0]()) == 0 or Int(c.dim[0]()) == 0:
        return

    var M = Int(a.dim[0]())
    comptime num_experts = b_packed.static_shape[0]
    comptime static_N = b_packed.static_shape[1]
    comptime static_K_packed = b_packed.static_shape[2]
    comptime static_K = static_K_packed * 2
    comptime fp8_type = DType.float8_e4m3fn

    # TODO: This implementation materializes the full FP8 weights and casted
    # activations into global memory before dispatching the grouped GEMM, which
    # negates the memory bandwidth benefits of MXFP4. Replace with a fused SM90
    # prologue that unpacks MXFP4 directly in shared memory or registers.

    # Step 1: Dequantize MXFP4 weights to FP8 for all experts.
    # Flatten [num_experts, N, K//2] -> [num_experts*N, K//2] for dequant,
    # then use [num_experts, N, K] for the grouped GEMM.
    var b_fp8_buf = ctx.enqueue_create_buffer[fp8_type](
        num_experts * static_N * static_K
    )
    var b_fp8_flat_tt = TileTensor(
        b_fp8_buf,
        row_major((Idx[num_experts * static_N](), Idx[static_K]())),
    )

    # Flatten packed weights and scales to 2D for dequant kernel.
    var b_packed_flat_tt = TileTensor[b_type](
        b_packed.ptr,
        row_major((Idx[num_experts * static_N](), Idx[static_K_packed]())),
    )
    comptime static_scale_cols = b_scales.static_shape[2]
    var b_scales_flat_tt = TileTensor[b_scales_type](
        b_scales.ptr,
        row_major(
            (Idx[num_experts * static_N](), Idx[static_scale_cols]())
        ),
    )

    dequant_mxfp4(
        ctx,
        b_fp8_flat_tt,
        b_packed_flat_tt,
        b_scales_flat_tt,
        num_rows=num_experts * static_N,
        num_cols=static_K,
    )

    # Reshape dequantized weights to 3D [num_experts, N, K] for grouped GEMM.
    var b_fp8_tt = TileTensor[fp8_type](
        b_fp8_buf,
        row_major[num_experts, static_N, static_K](),
    )

    # Step 2: Cast BF16 activations to FP8.
    var a_fp8_buf = ctx.enqueue_create_buffer[fp8_type](M * static_K)
    var a_fp8_tt = TileTensor(a_fp8_buf, row_major((Idx(M), Idx[static_K]())))

    _cast_bf16_to_fp8(ctx, a_fp8_tt, a, M, static_K)

    # Step 3: FP8 grouped GEMM via grouped_matmul_sm90.
    comptime BN = _find_largest_bn_for_sm90_matmul[fp8_type, static_N]()
    comptime wgmma_shape = IndexList[3](64, BN, 16)

    grouped_matmul_sm90[
        wgmma_shape=wgmma_shape,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](
        c,
        a_fp8_tt,
        a_offsets,
        max_num_tokens_per_expert,
        b_fp8_tt,
        expert_ids,
        num_active_experts,
        ctx,
    )

    # Keep temp buffers alive through async GEMM enqueue.
    _ = b_fp8_buf^
    _ = a_fp8_buf^


def _cast_bf16_to_fp8(
    ctx: DeviceContext,
    output: TileTensor,
    input: TileTensor,
    num_rows: Int,
    num_cols: Int,
) raises:
    var out_tt = output.as_any_origin()
    var in_tt = input.as_any_origin()
    comptime assert out_tt.flat_rank == 2, "output must be rank 2"
    comptime assert in_tt.flat_rank == 2, "input must be rank 2"
    comptime assert out_tt.mut, "output must be mutable"

    @always_inline
    @__copy_capture(out_tt, in_tt)
    @parameter
    def cast_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank],):
        comptime assert rank == 2, "cast_fn only supports rank-2 tensors"
        var idx = rebind[IndexList[2]](idx_arg)
        var coord = Coord(idx)
        comptime assert in_tt.flat_rank >= coord.flat_rank
        comptime assert out_tt.flat_rank >= coord.flat_rank
        out_tt.store[width=width](
            coord,
            in_tt.load[width=width](coord).cast[out_tt.dtype](),
        )

    elementwise[cast_fn, simd_width_of[input.dtype](), target="gpu"](
        Index(num_rows, num_cols), ctx
    )
