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
"""MXFP4 tile loader for fused dequant+GEMM on SM90.

Provides a TileLoader-compatible struct that dequantizes packed MXFP4 weights
(uint8, 2 FP4 values per byte) with E8M0 block scales directly into FP8
shared memory tiles, bypassing the global memory roundtrip.

Each of the 128 producer threads handles one row of the BN×BK tile, processing
BK/ELEMENTS_PER_THREAD chunks of 8 elements each.
"""

from std.math import ceildiv
from std.gpu import thread_idx_int as thread_idx
from std.gpu.globals import WARPGROUP_SIZE
from std.gpu.memory import AddressSpace
from std.memory import bitcast

from layout import Layout, LayoutTensor
from layout.layout import coalesce
from layout.swizzle import Swizzle
from ....structuring import SMemTile, SMemBarrier
from .tile_loader import TileLoader
from linalg.fp4_utils import cast_uint_to_fp4e2m1


struct TileLoaderMXFP4Dequant[
    b_packed_layout: Layout,
    b_scales_layout: Layout,
    BN: Int,
    BK: Int,
](TileLoader):
    """Tile loader that dequantizes MXFP4→FP8 directly into shared memory.

    Instead of loading FP8 data from global memory, this loader:
    1. Loads packed uint8 (2 FP4 values per byte) from b_packed
    2. Loads E8M0 scale bytes from b_scales
    3. Unpacks FP4→float32 via LUT, applies scale, casts to FP8
    4. Stores FP8 values to shared memory with SWIZZLE_128B pattern

    All 128 producer threads participate (one row per thread for BN=128).

    Parameters:
        b_packed_layout: Layout of the packed weight tensor [N, K//2].
        b_scales_layout: Layout of the scale tensor [N, K//SF_VECTOR_SIZE].
        BN: Block tile N dimension (must be 128).
        BK: Block tile K dimension (must be 128 for FP8).
    """

    comptime _dtype = DType.float8_e4m3fn

    # Swizzle for SWIZZLE_128B with FP8 (1 byte per element):
    # row stride = 128 bytes = BK elements
    # swizzled_col = col ^ ((row & 7) << 4)
    comptime _swizzle = Swizzle(3, 4, 3)

    # Each thread processes 8 FP4 elements at a time (4 packed bytes → 8 FP8)
    comptime ELEMENTS_PER_THREAD = 8
    comptime BYTES_PER_THREAD = Self.ELEMENTS_PER_THREAD // 2
    comptime SF_VECTOR_SIZE = 32

    var b_packed: LayoutTensor[
        DType.uint8,
        Self.b_packed_layout,
        ImmutAnyOrigin,
        address_space=AddressSpace.GENERIC,
    ]
    var b_scales: LayoutTensor[
        DType.float8_e8m0fnu,
        Self.b_scales_layout,
        ImmutAnyOrigin,
        address_space=AddressSpace.GENERIC,
    ]

    @always_inline
    def __init__(
        out self,
        b_packed: LayoutTensor[
            DType.uint8,
            Self.b_packed_layout,
            ImmutAnyOrigin,
            address_space=AddressSpace.GENERIC,
        ],
        b_scales: LayoutTensor[
            DType.float8_e8m0fnu,
            Self.b_scales_layout,
            ImmutAnyOrigin,
            address_space=AddressSpace.GENERIC,
        ],
    ):
        self.b_packed = b_packed
        self.b_scales = b_scales

    @always_inline
    def load_tile(
        self,
        dst: SMemTile[DType.float8_e4m3fn, _, alignment=128, ...],
        mem_barrier: SMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """Dequantize an MXFP4 tile and store FP8 result to shared memory.

        Args:
            dst: Destination FP8 tile in shared memory [BN, BK] with swizzle.
            mem_barrier: Memory barrier (signaling handled by CPAsyncBarrierHandler).
            coords: (n_tile_idx, k_tile_idx) - tile indices in the B matrix.
        """
        # coords: (n_tile_coord, k_tile_coord) where k is in BK-sized tiles
        var n_base = coords[0]  # Row (N dimension) base element offset
        var k_base = coords[1] * Self.BK  # K dimension base element offset

        # Each thread handles one row of the BN×BK tile
        var thread_row = thread_idx.x % Self.BN  # row within tile [0, BN)
        var n_global = n_base + thread_row

        # Check N bounds
        var n_dim = self.b_packed.dim[0]()
        if n_global >= n_dim:
            # Out of bounds: zero-fill this thread's row in smem
            _zero_fill_row(dst, thread_row)
            return

        # Get raw pointer to smem for swizzled stores
        var smem_ptr = dst.ptr

        # Process BK elements in chunks of ELEMENTS_PER_THREAD
        var k_dim_packed = self.b_packed.dim[1]()  # K//2

        comptime num_chunks = Self.BK // Self.ELEMENTS_PER_THREAD
        comptime for chunk_idx in range(num_chunks):
            var k_local = chunk_idx * Self.ELEMENTS_PER_THREAD
            var k_global = k_base + k_local
            var packed_col = k_global // 2

            # Bounds check on K dimension
            if packed_col + Self.BYTES_PER_THREAD > k_dim_packed:
                # Zero-fill remaining
                comptime for elem in range(Self.ELEMENTS_PER_THREAD):
                    var col = k_local + elem
                    var swizzled_offset = _swizzle_offset(thread_row, col)
                    (smem_ptr + swizzled_offset).store(
                        Scalar[DType.float8_e4m3fn](0)
                    )
                continue

            # Load 4 packed uint8 bytes (8 FP4 values)
            var packed_bytes = SIMD[DType.uint8, Self.BYTES_PER_THREAD]()
            comptime for b in range(Self.BYTES_PER_THREAD):
                packed_bytes[b] = self.b_packed[n_global, packed_col + b]

            # Unpack FP4 E2M1 → float32 via LUT
            var fp32_values = cast_uint_to_fp4e2m1[
                out_dtype=DType.float32,
                out_width=Self.ELEMENTS_PER_THREAD,
            ](packed_bytes)

            # Load E8M0 scale for this block of 32 elements
            var scale_col = k_global // Self.SF_VECTOR_SIZE
            var scale_e8m0 = self.b_scales[n_global, scale_col]

            # Convert E8M0 to float32
            var scale_f32 = scale_e8m0.cast[DType.float32]()

            # Apply scale and cast to FP8
            var scaled = fp32_values * scale_f32
            var fp8_values = scaled.cast[DType.float8_e4m3fn]()

            # Store to shared memory with SWIZZLE_128B pattern
            # For FP8 with BK=128: swizzled_col = col ^ ((row & 7) << 4)
            comptime for elem in range(Self.ELEMENTS_PER_THREAD):
                var col = k_local + elem
                var swizzled_offset = _swizzle_offset(thread_row, col)
                (smem_ptr + swizzled_offset).store(fp8_values[elem])


@always_inline
def _swizzle_offset(row: Int, col: Int) -> Int:
    """Compute swizzled linear offset for SWIZZLE_128B with FP8 (BK=128).

    For FP8 (1 byte per element), BK=128 → 128 bytes per row.
    SWIZZLE_128B: swizzled_col = col ^ ((row & 7) << 4)
    Linear offset = row * 128 + swizzled_col
    """
    var swizzled_col = col ^ ((row & 7) << 4)
    return row * 128 + swizzled_col


@always_inline
def _zero_fill_row(
    dst: SMemTile[DType.float8_e4m3fn, _, alignment=128, ...],
    row: Int,
):
    """Zero-fill an entire row of the smem tile."""
    var smem_ptr = dst.ptr
    comptime for col in range(128):
        var offset = _swizzle_offset(row, col)
        (smem_ptr + offset).store(Scalar[DType.float8_e4m3fn](0))
