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
"""MXFP4 matmul on H100 (SM90) with fused dequant + FP8 GEMM.

Casts BF16 activations to FP8 in global memory, then runs a fused kernel
that dequantizes MXFP4 weights directly into shared memory within the GEMM
producer pipeline, eliminating the global memory roundtrip for weights.
"""

from std.algorithm.functional import elementwise
from std.gpu.host import DeviceContext
from std.sys.info import _accelerator_arch, simd_width_of
from layout import Coord, Idx, TileTensor, row_major
from std.utils.index import Index, IndexList

from .matmul.gpu.sm90.mxfp4_fused_matmul import mxfp4_fused_gemm_sm90


def mxfp4_matmul_sm90(
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b_packed: TileTensor,
    b_scales: TileTensor,
    ctx: DeviceContext,
) raises:
    """MXFP4 matmul: fused dequant B in smem + FP8 GEMM on SM90.

    Casts BF16 activations to FP8, then runs the fused MXFP4 kernel that
    dequantizes weights directly into shared memory during the GEMM.

    Args:
        c: Output [M, N] in bfloat16.
        a: Activations [M, K] in bfloat16.
        b_packed: Weights [N, K//2] in uint8 (packed MXFP4).
        b_scales: Weight scales [N, K//32] in float8_e8m0fnu.
        ctx: Device context.
    """
    comptime assert (
        "sm_90" in _accelerator_arch()
    ), "mxfp4_matmul_sm90 requires SM90"

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

    var M = Int(c.dim[0]())
    comptime static_N = type_of(c).static_shape[1]
    comptime static_K = type_of(a).static_shape[1]
    comptime fp8_type = DType.float8_e4m3fn

    # Step 1: Cast BF16 activations to FP8
    var a_fp8_buf = ctx.enqueue_create_buffer[fp8_type](M * static_K)
    var a_fp8_tt = TileTensor(a_fp8_buf, row_major((Idx(M), Idx[static_K]())))

    _cast_bf16_to_fp8(ctx, a_fp8_tt, a, M, static_K)

    # Step 2: Fused MXFP4 dequant + FP8 GEMM (B dequant happens in smem)
    mxfp4_fused_gemm_sm90[
        c_type,
        fp8_type,
        N=static_N,
        K=static_K,
    ](c, a_fp8_tt, b_packed, b_scales, ctx)

    # Keep temp buffer alive through async kernel enqueue.
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
