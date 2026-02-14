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

"""Fat tensor type for quantized weights with embedded scale factors."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
)
from max.graph.type import Shape
from max.nn.legacy.float8_config import (
    Float8Config,
    nvfp4_packed_k,
)
from max.nn.legacy.float8_ops import matmul_float4, matmul_float8
from max.support.math import ceildiv

from .layer import Module, Shardable


class FatTensor(Module, Shardable):
    """A quantized tensor that bundles weight data with its scale factors.

    Unlike plain :obj:`Weight` objects which are data containers (PoD),
    ``FatTensor`` is a self-contained quantized tensor that tracks its own
    scales and supports fp4 (NVFP4) and fp8 quantization natively.

    A ``FatTensor`` encapsulates:

    - The quantized weight data
    - Weight scale factor(s) (shape depends on granularity)
    - Optional input scale factor (for static quantization)
    - Optional secondary weight scale (for NVFP4)
    - The :obj:`Float8Config` describing the quantization scheme

    When called, it performs a quantization-aware matrix multiplication.

    Example:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef
        from max.nn.legacy import FatTensor

        qweight = FatTensor(
            in_dim=4096,
            out_dim=4096,
            dtype=DType.float8_e4m3fn,
            device=DeviceRef.GPU(),
            float8_config=float8_config,
        )

        # In forward pass:
        result = qweight(x)  # quantization-aware x @ weight.T
    """

    weight: Weight
    """The quantized weight data."""

    weight_scale: Weight
    """Weight scale factor(s). Shape depends on the scale granularity
    (scalar for per-tensor, ``(N, 1)`` for row-wise, ``(ceil(N/B0), ceil(K/B1))``
    for block-wise)."""

    input_scale: Weight | None
    """Input scale factor for static quantization. ``None`` when using
    dynamic scaling."""

    weight_scale_2: Weight | None
    """Secondary weight scale factor, only used for NVFP4 quantization."""

    float8_config: Float8Config
    """The quantization configuration."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        float8_config: Float8Config,
        *,
        _is_sharding: bool = False,
    ) -> None:
        """Initializes a FatTensor with quantized weight data and scales.

        The weight data, weight scale, and optional input/secondary scales
        are created as :obj:`Weight` objects. The scale shapes are inferred
        from the :obj:`Float8Config`.

        Args:
            in_dim: The input dimension of the weight matrix.
            out_dim: The output dimension of the weight matrix.
            dtype: The :obj:`DType` of the quantized weight data (e.g.
                ``DType.float8_e4m3fn``, ``DType.uint8`` for fp4).
            device: The target :obj:`DeviceRef` for the weight data.
            float8_config: The :obj:`Float8Config` describing the
                quantization scheme.
            _is_sharding: Internal flag to skip weight creation during
                sharding.
        """
        super().__init__()
        self.float8_config = float8_config
        self.device = device
        self.input_scale = None
        self.weight_scale_2 = None

        if not _is_sharding:
            packed_in_dim = nvfp4_packed_k(in_dim, float8_config)

            self.weight = Weight(
                name="weight",
                dtype=dtype,
                shape=(out_dim, packed_in_dim),
                device=device,
            )

            weight_scale_shape = _infer_weight_scale_shape(
                float8_config, out_dim, packed_in_dim
            )
            self.weight_scale = Weight(
                name="weight_scale",
                dtype=float8_config.weight_scale.dtype,
                shape=weight_scale_shape,
                device=DeviceRef.CPU(),
            )

            if float8_config.is_static:
                self.input_scale = Weight(
                    name="input_scale",
                    dtype=float8_config.input_scale.dtype,
                    shape=(),
                    device=DeviceRef.CPU(),
                )

            if float8_config.is_nvfp4:
                self.weight_scale_2 = Weight(
                    name="weight_scale_2",
                    dtype=float8_config.input_scale.dtype,
                    shape=(),
                    device=DeviceRef.CPU(),
                )

    @property
    def is_fp4(self) -> bool:
        """Whether this tensor uses fp4 (NVFP4) quantization."""
        return self.float8_config.is_nvfp4

    @property
    def is_fp8(self) -> bool:
        """Whether this tensor uses fp8 quantization (not fp4)."""
        return not self.float8_config.is_nvfp4

    def __call__(self, x: TensorValue) -> TensorValue:
        """Performs quantization-aware matmul: ``x @ self.weight.T``.

        Dispatches to the appropriate quantized matmul implementation
        based on the quantization config (fp4 vs fp8, static vs dynamic
        scaling).

        Args:
            x: Input :obj:`TensorValue`. The last dimension must match
                the weight's input dimension.

        Returns:
            The result of the quantization-aware matrix multiplication.
        """
        if self.is_fp4:
            assert self.input_scale is not None
            assert self.weight_scale_2 is not None
            return matmul_float4(
                x,
                self.weight,
                self.weight_scale,
                self.input_scale,
                self.weight_scale_2,
                self.float8_config,
            )
        else:
            return matmul_float8(
                x,
                self.weight,
                self.weight_scale,
                self.input_scale,
                self.float8_config,
            )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the weight sharding strategy."""
        return self.weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the sharding strategy for weight and scales together.

        The weight scale sharding is determined automatically based on
        the scale granularity and weight sharding strategy:

        - Per-tensor scales are always replicated.
        - Row-wise scales follow the weight strategy unless the weight
          is column-sharded, in which case they are replicated.
        - Block-wise scales are sharded along the corresponding
          dimension.

        Args:
            strategy: The :obj:`ShardingStrategy` to apply.
        """
        self.weight.sharding_strategy = strategy

        # Determine weight scale sharding based on granularity + weight strategy.
        ws = self.float8_config.weight_scale
        if ws.is_tensor:
            self.weight_scale.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        elif ws.is_rowwise:
            if strategy.is_colwise or strategy.is_head_aware_colwise:
                self.weight_scale.sharding_strategy = (
                    ShardingStrategy.replicate(strategy.num_devices)
                )
            else:
                self.weight_scale.sharding_strategy = strategy
        elif ws.is_block:
            if strategy.is_rowwise:
                self.weight_scale.sharding_strategy = strategy
            elif strategy.is_colwise or strategy.is_head_aware_colwise:
                if strategy.is_head_aware_colwise:
                    assert isinstance(strategy.shard, partial)
                    num_heads = strategy.shard.keywords["num_heads"]
                    head_dim = strategy.shard.keywords["head_dim"]
                    assert ws.block_size is not None
                    block_size_k = ws.block_size[1]
                    if head_dim % block_size_k == 0:
                        scale_head_dim = head_dim // block_size_k
                        self.weight_scale.sharding_strategy = (
                            ShardingStrategy.head_aware_columnwise(
                                strategy.num_devices,
                                num_heads,
                                scale_head_dim,
                            )
                        )
                    else:
                        self.weight_scale.sharding_strategy = (
                            ShardingStrategy.columnwise(strategy.num_devices)
                        )
                else:
                    self.weight_scale.sharding_strategy = (
                        ShardingStrategy.columnwise(strategy.num_devices)
                    )
            else:
                self.weight_scale.sharding_strategy = strategy
        else:
            self.weight_scale.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[FatTensor]:
        """Creates sharded views of this FatTensor across multiple devices.

        Both weight data and scale tensors are sharded according to the
        sharding strategy set via :attr:`sharding_strategy`. Scalar scales
        (per-tensor, input scale, weight_scale_2) are shared across shards.

        Args:
            devices: Iterable of :obj:`DeviceRef` devices to shard across.

        Returns:
            List of sharded :obj:`FatTensor` instances, one per device.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if not self.weight.sharding_strategy:
            raise ValueError(
                "FatTensor cannot be sharded because no sharding strategy "
                "was provided."
            )

        devices_list = list(devices)
        sharded_weights = self.weight.shard(devices_list)

        # Shard weight scale if non-scalar, otherwise share the reference.
        sharded_weight_scales = (
            self.weight_scale.shard(devices_list)
            if len(self.weight_scale.shape) > 0
            else None
        )

        shards = []
        for shard_idx, (device, weight_shard) in enumerate(
            zip(devices_list, sharded_weights, strict=True)
        ):
            shard = FatTensor(
                in_dim=int(self.weight.shape[1]),
                out_dim=int(self.weight.shape[0]),
                dtype=self.weight.dtype,
                device=device,
                float8_config=self.float8_config,
                _is_sharding=True,
            )

            shard.weight = weight_shard

            if sharded_weight_scales is not None:
                shard.weight_scale = sharded_weight_scales[shard_idx]
            else:
                shard.weight_scale = self.weight_scale

            # Input scale is always scalar — share across shards.
            shard.input_scale = self.input_scale

            # weight_scale_2 is always scalar (NVFP4) — share across shards.
            if self.weight_scale_2 is not None:
                shard.weight_scale_2 = self.weight_scale_2

            shards.append(shard)

        return shards

    def __repr__(self) -> str:
        quant_type = "fp4" if self.is_fp4 else "fp8"
        return (
            f"FatTensor({quant_type}, dtype={self.weight.dtype}, "
            f"shape={self.weight.shape}, device={self.device})"
        )


def _infer_weight_scale_shape(
    float8_config: Float8Config,
    out_dim: int,
    packed_in_dim: int,
) -> tuple[int, ...]:
    """Computes the weight scale shape from the config and weight dimensions.

    Args:
        float8_config: The quantization configuration.
        out_dim: The output dimension (N) of the weight.
        packed_in_dim: The input dimension (K), possibly packed for NVFP4.

    Returns:
        The shape of the weight scale tensor.
    """
    ws = float8_config.weight_scale
    if ws.is_rowwise:
        return (out_dim, 1)
    elif ws.is_tensor:
        return ()
    elif ws.is_block:
        assert ws.block_size is not None
        return (
            ceildiv(out_dim, ws.block_size[0]),
            ceildiv(packed_in_dim, ws.block_size[1]),
        )
    else:
        raise ValueError(
            f"Unsupported weight scale granularity: {ws.granularity}"
        )
