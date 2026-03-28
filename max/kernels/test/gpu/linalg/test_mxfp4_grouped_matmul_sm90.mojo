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
"""Smoke test for MXFP4 grouped matmul on H100 (SM90).

Validates mxfp4_grouped_matmul_sm90 by comparing GPU output against a naive
CPU reference for several expert configurations and token distributions.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from linalg.mxfp4_grouped_matmul_sm90 import mxfp4_grouped_matmul_sm90
from linalg.fp4_utils import E2M1_TO_FLOAT32
from std.testing import assert_almost_equal
from std.utils.index import Index, IndexList


def _pack_fp4_pair(low: UInt8, high: UInt8) -> UInt8:
    """Packs two 4-bit FP4 values into one uint8 byte."""
    return (high & UInt8(0x0F)) << UInt8(4) | (low & UInt8(0x0F))


def _e8m0_to_float32(bits: UInt8) -> Float32:
    """Converts float8_e8m0fnu scale byte to float32: 2^(exp-127)."""
    if bits == UInt8(0):
        return Float32(0.0)
    var f32_bits = UInt32(bits) << UInt32(23)
    return bitcast[DType.float32](f32_bits)


def _cpu_grouped_matmul_mxfp4[
    num_experts: Int,
    N: Int,
    K: Int,
](
    c_host: UnsafePointer[mut=True, Scalar[DType.bfloat16]],
    a_host: UnsafePointer[mut=False, Scalar[DType.bfloat16]],
    b_packed_host: UnsafePointer[mut=False, Scalar[DType.uint8]],
    b_scales_host: UnsafePointer[mut=False, Scalar[DType.uint8]],
    a_offsets: List[Int],
    expert_ids: List[Int],
    num_active_experts: Int,
):
    """CPU reference: grouped matmul with MXFP4 weights.

    C[a_offsets[i]:a_offsets[i+1], :] = A[a_offsets[i]:a_offsets[i+1], :] @ B[expert_ids[i], :, :].T
    where B is stored in MXFP4 format.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    for group_idx in range(num_active_experts):
        var start_row = a_offsets[group_idx]
        var end_row = a_offsets[group_idx + 1]
        var expert = expert_ids[group_idx]

        for m in range(start_row, end_row):
            for n in range(N):
                var accum = Float32(0.0)

                for k in range(K):
                    # Load activation
                    var a_val = a_host[m * K + k].cast[DType.float32]()

                    # Dequantize MXFP4 weight
                    var b_offset = expert * N * packed_K + n * packed_K + k // 2
                    var packed_byte = b_packed_host[b_offset]
                    var nibble_shift = UInt8((k % 2) * 4)
                    var fp4_bits = Int(
                        (packed_byte >> nibble_shift) & UInt8(0x0F)
                    )
                    var fp32_val = E2M1_TO_FLOAT32[fp4_bits]

                    # Apply E8M0 scale
                    var scale_offset = (
                        expert * N * scale_K + n * scale_K + k // 32
                    )
                    var scale_byte = b_scales_host[scale_offset]
                    var scale_f32 = _e8m0_to_float32(scale_byte)

                    var b_val = fp32_val * scale_f32
                    accum += a_val * b_val

                c_host[m * N + n] = accum.cast[DType.bfloat16]()


def test_mxfp4_grouped_matmul[
    num_experts: Int,
    N: Int,
    K: Int,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
    scale_exp: UInt8 = UInt8(127),
) raises:
    """Test MXFP4 grouped matmul against CPU reference."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    # Compute total tokens and max tokens per expert.
    var total_tokens = 0
    var max_tokens_per_expert = 0
    for i in range(len(num_tokens_by_expert)):
        total_tokens += num_tokens_by_expert[i]
        max_tokens_per_expert = max(
            max_tokens_per_expert, num_tokens_by_expert[i]
        )

    print(
        "  num_experts=",
        num_experts,
        " N=",
        N,
        " K=",
        K,
        " active=",
        num_active_experts,
        " total_tokens=",
        total_tokens,
    )

    # Allocate host buffers.
    var a_size = total_tokens * K
    var c_size = total_tokens * N
    var b_packed_size = num_experts * N * packed_K
    var b_scales_size = num_experts * N * scale_K

    var a_host = alloc[Scalar[DType.bfloat16]](a_size)
    var c_host = alloc[Scalar[DType.bfloat16]](c_size)
    var c_ref_host = alloc[Scalar[DType.bfloat16]](c_size)
    var b_packed_host = alloc[UInt8](b_packed_size)
    var b_scales_host = alloc[UInt8](b_scales_size)
    var a_offsets_host = alloc[Scalar[DType.uint32]](num_experts + 1)
    var expert_ids_host = alloc[Scalar[DType.int32]](num_experts)

    # Fill activations with small random-ish values.
    for i in range(a_size):
        # Use a simple deterministic pattern that stays in bf16 range.
        a_host[i] = Scalar[DType.bfloat16](
            Float32((i % 17) - 8) * Float32(0.1)
        )

    # Fill packed MXFP4 weights with a deterministic pattern.
    for i in range(b_packed_size):
        var low = UInt8((i * 3) % 16)
        var high = UInt8((i * 3 + 7) % 16)
        b_packed_host[i] = _pack_fp4_pair(low, high)

    # Fill scales with the given scale exponent.
    for i in range(b_scales_size):
        b_scales_host[i] = scale_exp

    # Set up offsets and expert ids.
    a_offsets_host[0] = 0
    for i in range(num_active_experts):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids_list[i])

    # Build offsets list for CPU reference.
    var offsets_list = List[Int]()
    for i in range(num_active_experts + 1):
        offsets_list.append(Int(a_offsets_host[i]))

    # CPU reference.
    _cpu_grouped_matmul_mxfp4[num_experts, N, K](
        c_ref_host,
        a_host,
        b_packed_host,
        b_scales_host,
        offsets_list,
        expert_ids_list,
        num_active_experts,
    )

    # Create device buffers.
    var a_dev_buf = ctx.enqueue_create_buffer[DType.bfloat16](a_size)
    var c_dev_buf = ctx.enqueue_create_buffer[DType.bfloat16](c_size)
    var b_packed_dev_buf = ctx.enqueue_create_buffer[DType.uint8](
        b_packed_size
    )
    var b_scales_dev_buf = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        b_scales_size
    )
    var a_offsets_dev_buf = ctx.enqueue_create_buffer[DType.uint32](
        num_experts + 1
    )
    var expert_ids_dev_buf = ctx.enqueue_create_buffer[DType.int32](
        num_experts
    )

    # Copy to device.
    ctx.enqueue_copy(a_dev_buf, a_host)
    ctx.enqueue_copy(b_packed_dev_buf, b_packed_host)

    # Copy scales (need to reinterpret uint8 -> float8_e8m0fnu).
    var b_scales_host_buf = ctx.enqueue_create_host_buffer[
        DType.float8_e8m0fnu
    ](b_scales_size)
    for i in range(b_scales_size):
        b_scales_host_buf.unsafe_ptr()[i] = rebind[
            Scalar[DType.float8_e8m0fnu]
        ](b_scales_host[i])
    ctx.enqueue_copy(b_scales_dev_buf, b_scales_host_buf)

    ctx.enqueue_copy(a_offsets_dev_buf, a_offsets_host)
    ctx.enqueue_copy(expert_ids_dev_buf, expert_ids_host)
    ctx.synchronize()

    # Create TileTensors.
    var a_dev = TileTensor[DType.bfloat16](
        a_dev_buf,
        row_major(Coord(Idx(total_tokens), Idx[K]())),
    )
    var c_dev = TileTensor[DType.bfloat16](
        c_dev_buf,
        row_major(Coord(Idx(total_tokens), Idx[N]())),
    )
    var b_packed_dev = TileTensor[DType.uint8](
        b_packed_dev_buf,
        row_major[num_experts, N, packed_K](),
    )
    var b_scales_dev = TileTensor[DType.float8_e8m0fnu](
        b_scales_dev_buf,
        row_major[num_experts, N, scale_K](),
    )
    var a_offsets_dev = TileTensor[DType.uint32](
        a_offsets_dev_buf,
        row_major(Coord(Idx(num_experts + 1))),
    )
    var expert_ids_dev = TileTensor[DType.int32](
        expert_ids_dev_buf,
        row_major(Coord(Idx[num_experts]())),
    )

    # Run the kernel.
    mxfp4_grouped_matmul_sm90(
        c_dev,
        a_dev,
        b_packed_dev,
        b_scales_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_tokens_per_expert,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    # Copy back result.
    ctx.enqueue_copy(c_host, c_dev_buf)
    ctx.synchronize()

    # Compare with CPU reference.
    # FP8 intermediate precision means we need a generous tolerance.
    var rtol = Float64(0.15)
    var atol = Float64(0.5)
    var num_mismatches = 0
    for i in range(c_size):
        var got = c_host[i].cast[DType.float32]()
        var exp = c_ref_host[i].cast[DType.float32]()
        var err = abs(got - exp)
        var rel_err = err / max(abs(exp), Float32(1e-6))
        if err > atol and rel_err > rtol:
            if num_mismatches < 5:
                var m = i // N
                var n = i % N
                print(
                    "    MISMATCH [",
                    m,
                    ",",
                    n,
                    "]: got=",
                    got,
                    " expected=",
                    exp,
                    " err=",
                    err,
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL: ",
            num_mismatches,
            "/",
            c_size,
            " mismatches",
        )
        raise "MXFP4 grouped matmul test failed"
    else:
        print("    PASS")

    # Cleanup.
    a_host.free()
    c_host.free()
    c_ref_host.free()
    b_packed_host.free()
    b_scales_host.free()
    a_offsets_host.free()
    expert_ids_host.free()
    _ = a_dev_buf^
    _ = c_dev_buf^
    _ = b_packed_dev_buf^
    _ = b_scales_dev_buf^
    _ = a_offsets_dev_buf^
    _ = expert_ids_dev_buf^
    _ = b_scales_host_buf^


def main() raises:
    with DeviceContext() as ctx:
        print("=== MXFP4 Grouped MatMul SM90 Tests ===")

        # Single expert, single group.
        test_mxfp4_grouped_matmul[num_experts=1, N=256, K=256](
            1, [128], [0], ctx
        )

        # Single expert, small shape.
        test_mxfp4_grouped_matmul[num_experts=1, N=64, K=64](
            1, [32], [0], ctx
        )

        # Multiple experts, single active.
        test_mxfp4_grouped_matmul[num_experts=4, N=256, K=128](
            1, [64], [2], ctx
        )

        # Multiple experts, multiple active groups.
        test_mxfp4_grouped_matmul[num_experts=4, N=256, K=128](
            3, [32, 64, 16], [0, 2, 3], ctx
        )

        # Unaligned token counts.
        test_mxfp4_grouped_matmul[num_experts=4, N=256, K=256](
            2, [13, 7], [1, 3], ctx
        )

        # Larger shape closer to real MoE dims.
        test_mxfp4_grouped_matmul[num_experts=8, N=512, K=256](
            4, [32, 16, 48, 24], [0, 3, 5, 7], ctx
        )

        print("=== All tests passed ===")
