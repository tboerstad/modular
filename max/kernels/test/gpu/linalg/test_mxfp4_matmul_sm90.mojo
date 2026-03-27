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
"""End-to-end test for fused MXFP4 matmul on SM90 (H100).

Tests mxfp4_matmul_sm90 by:
1. Creating BF16 activations and MXFP4 packed weights with known patterns
2. Computing CPU reference: dequant MXFP4→float32, matmul in float32, cast→BF16
3. Running the GPU fused kernel
4. Comparing GPU output against CPU reference with tolerance for FP8 precision
"""

from std.math import ceildiv
from std.memory import bitcast
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major
from layout.coord import Coord, Idx
from linalg.mxfp4_matmul_sm90 import mxfp4_matmul_sm90
from linalg.fp4_utils import E2M1_TO_FLOAT32


def _pack_fp4_pair(low: UInt8, high: UInt8) -> UInt8:
    """Packs two 4-bit FP4 values into one uint8 byte."""
    return (high & UInt8(0x0F)) << UInt8(4) | (low & UInt8(0x0F))


def _e8m0_to_float32(bits: UInt8) -> Float32:
    """Converts float8_e8m0fnu scale byte to float32: 2^(exp-127)."""
    if bits == UInt8(0):
        return Float32(0.0)
    var f32_bits = UInt32(bits) << UInt32(23)
    return bitcast[DType.float32](f32_bits)


def _cpu_mxfp4_matmul[
    M: Int, N: Int, K: Int
](
    c_host: UnsafePointer[mut=True, Scalar[DType.bfloat16], _],
    a_host: UnsafePointer[mut=False, Scalar[DType.bfloat16], _],
    b_packed_host: UnsafePointer[mut=False, Scalar[DType.uint8], _],
    b_scales_host: UnsafePointer[mut=False, Scalar[DType.uint8], _],
):
    """CPU reference: dequant MXFP4 weights, then matmul C = A @ B^T.

    A is [M, K] in BF16, B (transposed) is [N, K] as MXFP4.
    C is [M, N] in BF16.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    for m in range(M):
        for n in range(N):
            var acc = Float32(0.0)
            for k in range(K):
                # Get A value
                var a_val = a_host[m * K + k].cast[DType.float32]()

                # Dequant B[n, k] from MXFP4
                var packed_col = k // 2
                var packed_byte = b_packed_host[n * packed_K + packed_col]
                var nibble_shift = UInt8((k % 2) * 4)
                var fp4_bits = Int(
                    (packed_byte >> nibble_shift) & UInt8(0x0F)
                )
                var b_fp32 = E2M1_TO_FLOAT32[fp4_bits]

                var scale_col = k // 32
                var scale_byte = b_scales_host[n * scale_K + scale_col]
                var scale_f32 = _e8m0_to_float32(scale_byte)

                var b_val = b_fp32 * scale_f32

                acc += a_val * b_val

            c_host[m * N + n] = acc.cast[DType.bfloat16]()


def test_mxfp4_matmul[
    M: Int, N: Int, K: Int
](ctx: DeviceContext, scale_exp: UInt8) raises:
    """Tests fused MXFP4 matmul for given shape and scale."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    # FP8 intermediate precision loses accuracy; use wider tolerance
    comptime tol = Float32(0.5)

    var scale_f32 = _e8m0_to_float32(scale_exp)
    print(
        "  M=", M, " N=", N, " K=", K, " scale_exp=", scale_exp,
        " (scale=", scale_f32, ")",
    )

    comptime a_size = M * K
    comptime b_packed_size = N * packed_K
    comptime b_scales_size = N * scale_K
    comptime c_size = M * N

    # Allocate host data
    var a_host = alloc[Scalar[DType.bfloat16]](a_size)
    var b_packed_host = alloc[UInt8](b_packed_size)
    var b_scales_host = alloc[UInt8](b_scales_size)
    var expected_host = alloc[Scalar[DType.bfloat16]](c_size)

    # Fill activations with small values to avoid FP8 overflow
    for m in range(M):
        for k in range(K):
            # Use a simple repeating pattern: values in [-1, 1]
            var val = Float32((m * K + k) % 7 - 3) * Float32(0.25)
            a_host[m * K + k] = val.cast[DType.bfloat16]()

    # Fill packed weights with deterministic FP4 pattern
    for n in range(N):
        for col in range(packed_K):
            var low = UInt8((col * 2) % 8)  # Use values 0-7 (positive E2M1)
            var high = UInt8((col * 2 + 1) % 8)
            b_packed_host[n * packed_K + col] = _pack_fp4_pair(low, high)

    # Fill scales
    for i in range(b_scales_size):
        b_scales_host[i] = scale_exp

    # CPU reference
    _cpu_mxfp4_matmul[M, N, K](
        expected_host, a_host, b_packed_host, b_scales_host
    )

    # Device buffers
    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](a_size)
    var b_packed_device = ctx.enqueue_create_buffer[DType.uint8](b_packed_size)
    var b_scales_device = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        b_scales_size
    )
    var c_device = ctx.enqueue_create_buffer[DType.bfloat16](c_size)

    # Upload inputs
    var a_host_buf = ctx.enqueue_create_host_buffer[DType.bfloat16](a_size)
    var b_packed_host_buf = ctx.enqueue_create_host_buffer[DType.uint8](
        b_packed_size
    )
    var b_scales_host_buf = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        b_scales_size
    )

    for i in range(a_size):
        a_host_buf.unsafe_ptr()[i] = a_host[i]
    for i in range(b_packed_size):
        b_packed_host_buf.unsafe_ptr()[i] = b_packed_host[i]
    for i in range(b_scales_size):
        b_scales_host_buf.unsafe_ptr()[i] = rebind[Scalar[DType.float8_e8m0fnu]](
            b_scales_host[i]
        )

    ctx.enqueue_copy(a_device, a_host_buf)
    ctx.enqueue_copy(b_packed_device, b_packed_host_buf)
    ctx.enqueue_copy(b_scales_device, b_scales_host_buf)
    ctx.synchronize()

    # Create TileTensors
    var a_tt = TileTensor(a_device, row_major[M, K]())
    var b_packed_tt = TileTensor(
        b_packed_device, row_major[N, packed_K]()
    )
    var b_scales_tt = TileTensor(
        b_scales_device, row_major[N, scale_K]()
    )
    var c_tt = TileTensor(c_device, row_major[M, N]())

    # Run fused MXFP4 matmul
    mxfp4_matmul_sm90(c_tt, a_tt, b_packed_tt, b_scales_tt, ctx)
    ctx.synchronize()

    # Copy output back
    var c_host_buf = ctx.enqueue_create_host_buffer[DType.bfloat16](c_size)
    ctx.enqueue_copy(c_host_buf, c_device)
    ctx.synchronize()

    # Compare
    var max_err = Float32(0.0)
    var max_rel_err = Float32(0.0)
    var num_mismatches = 0
    for i in range(c_size):
        var got = c_host_buf.unsafe_ptr()[i].cast[DType.float32]()
        var exp = expected_host[i].cast[DType.float32]()
        var err = abs(got - exp)
        var rel_err = err / max(abs(exp), Float32(1e-6))
        max_err = max(max_err, err)
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > tol and err > Float32(0.1):
            if num_mismatches < 5:
                var row = i // N
                var col = i % N
                print(
                    "    MISMATCH [", row, ",", col, "]: got=", got,
                    " expected=", exp, " err=", err, " rel=", rel_err,
                )
            num_mismatches += 1

    a_host.free()
    b_packed_host.free()
    b_scales_host.free()
    expected_host.free()

    if num_mismatches > 0:
        print(
            "    FAIL: ", num_mismatches, " mismatches, max_err=", max_err,
            " max_rel_err=", max_rel_err,
        )
        raise Error("MXFP4 fused matmul test failed")

    print("    PASS max_err=", max_err, " max_rel_err=", max_rel_err)


def main() raises:
    with DeviceContext() as ctx:
        print("MXFP4 Fused Matmul SM90 Tests")
        print("==============================")

        # Small shapes (basic correctness)
        print("-- Small shapes, scale=1.0 --")
        test_mxfp4_matmul[128, 128, 128](ctx, UInt8(127))
        test_mxfp4_matmul[128, 256, 256](ctx, UInt8(127))

        # Medium shapes
        print("-- Medium shapes, scale=1.0 --")
        test_mxfp4_matmul[256, 256, 512](ctx, UInt8(127))
        test_mxfp4_matmul[512, 512, 512](ctx, UInt8(127))

        # Scale = 2.0 (exponent 128)
        print("-- Scale = 2.0 --")
        test_mxfp4_matmul[128, 128, 128](ctx, UInt8(128))
        test_mxfp4_matmul[256, 256, 256](ctx, UInt8(128))

        # Scale = 0.5 (exponent 126)
        print("-- Scale = 0.5 --")
        test_mxfp4_matmul[128, 256, 256](ctx, UInt8(126))

        # Larger shape (representative of MoE dimensions)
        print("-- Larger shape --")
        test_mxfp4_matmul[256, 512, 1024](ctx, UInt8(127))

        print("==============================")
        print("ALL TESTS PASSED")
