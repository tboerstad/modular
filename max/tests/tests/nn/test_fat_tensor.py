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
"""Tests for FatTensor in max.nn.legacy."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType
from max.nn.legacy.fat_tensor import FatTensor
from max.nn.legacy.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)


def _make_fp8_config(
    *,
    weight_granularity: Float8ScaleGranularity = Float8ScaleGranularity.TENSOR,
    input_origin: Float8ScaleOrigin = Float8ScaleOrigin.DYNAMIC,
    input_granularity: Float8ScaleGranularity = Float8ScaleGranularity.TENSOR,
    block_size: tuple[int, int] | None = None,
) -> Float8Config:
    """Helper to create a Float8Config for tests."""
    return Float8Config(
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float32,
            granularity=weight_granularity,
            block_size=block_size,
        ),
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=input_granularity,
            origin=input_origin,
            block_size=(1, 128)
            if input_granularity == Float8ScaleGranularity.BLOCK
            else None,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
    )


def _make_nvfp4_config() -> Float8Config:
    """Helper to create an NVFP4 Float8Config for tests."""
    return Float8Config(
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float8_e4m3fn,
            granularity=Float8ScaleGranularity.BLOCK,
            block_size=(1, 16),
        ),
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.BLOCK,
            origin=Float8ScaleOrigin.DYNAMIC,
            block_size=(1, 16),
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
        quant_method="modelopt",
        quant_algo="NVFP4",
    )


# ---- Construction tests ----


def test_fat_tensor_fp8_dynamic_tensor_scale() -> None:
    """FatTensor with fp8, dynamic scaling, per-tensor weight scale."""
    config = _make_fp8_config()
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.GPU(),
        float8_config=config,
    )

    assert ft.is_fp8
    assert not ft.is_fp4
    assert ft.weight.dtype == DType.float8_e4m3fn
    assert tuple(int(d) for d in ft.weight.shape) == (1024, 4096)
    assert ft.weight_scale is not None
    assert len(ft.weight_scale.shape) == 0  # scalar
    assert ft.input_scale is None  # dynamic
    assert ft.weight_scale_2 is None


def test_fat_tensor_fp8_static_tensor_scale() -> None:
    """FatTensor with fp8, static scaling, per-tensor weight scale."""
    config = _make_fp8_config(input_origin=Float8ScaleOrigin.STATIC)
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.GPU(),
        float8_config=config,
    )

    assert ft.input_scale is not None
    assert len(ft.input_scale.shape) == 0  # scalar
    assert ft.input_scale.device == DeviceRef.CPU()


def test_fat_tensor_fp8_rowwise_weight_scale() -> None:
    """FatTensor with fp8, row-wise weight scale."""
    config = _make_fp8_config(
        weight_granularity=Float8ScaleGranularity.ROWWISE
    )
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.GPU(),
        float8_config=config,
    )

    assert ft.weight_scale is not None
    assert tuple(int(d) for d in ft.weight_scale.shape) == (1024, 1)


def test_fat_tensor_fp8_block_weight_scale() -> None:
    """FatTensor with fp8, block-wise weight scale."""
    config = _make_fp8_config(
        weight_granularity=Float8ScaleGranularity.BLOCK,
        input_granularity=Float8ScaleGranularity.BLOCK,
        block_size=(128, 128),
    )
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.GPU(),
        float8_config=config,
    )

    assert ft.weight_scale is not None
    # 1024/128 = 8, 4096/128 = 32
    assert tuple(int(d) for d in ft.weight_scale.shape) == (8, 32)


def test_fat_tensor_nvfp4() -> None:
    """FatTensor with NVFP4 quantization."""
    config = _make_nvfp4_config()
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.uint8,
        device=DeviceRef.GPU(),
        float8_config=config,
    )

    assert ft.is_fp4
    assert not ft.is_fp8
    # NVFP4 packs 2 values per byte
    assert tuple(int(d) for d in ft.weight.shape) == (1024, 2048)
    assert ft.weight_scale is not None
    assert ft.weight_scale_2 is not None
    assert len(ft.weight_scale_2.shape) == 0  # scalar


def test_fat_tensor_repr() -> None:
    """FatTensor repr includes quant type and shape."""
    config = _make_fp8_config()
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.GPU(),
        float8_config=config,
    )
    r = repr(ft)
    assert "fp8" in r
    assert "float8_e4m3fn" in r


def test_fat_tensor_nvfp4_repr() -> None:
    """FatTensor repr shows fp4 for NVFP4."""
    config = _make_nvfp4_config()
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.uint8,
        device=DeviceRef.GPU(),
        float8_config=config,
    )
    r = repr(ft)
    assert "fp4" in r


# ---- Module integration tests ----


def test_fat_tensor_weight_discovery() -> None:
    """FatTensor's weights are discoverable via Module's layer_weights."""
    config = _make_fp8_config(input_origin=Float8ScaleOrigin.STATIC)
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.GPU(),
        float8_config=config,
    )

    weights = ft.layer_weights
    assert "weight" in weights
    assert "weight_scale" in weights
    assert "input_scale" in weights


def test_fat_tensor_nvfp4_weight_discovery() -> None:
    """NVFP4 FatTensor's weights include weight_scale_2."""
    config = _make_nvfp4_config()
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.uint8,
        device=DeviceRef.GPU(),
        float8_config=config,
    )

    weights = ft.layer_weights
    assert "weight" in weights
    assert "weight_scale" in weights
    assert "weight_scale_2" in weights


def test_fat_tensor_as_sublayer() -> None:
    """FatTensor integrates as sublayer of another Module."""
    from max.nn.legacy.layer import Module

    class TestModule(Module):
        def __init__(self):
            super().__init__()
            config = _make_fp8_config(input_origin=Float8ScaleOrigin.STATIC)
            self.q_proj = FatTensor(
                in_dim=4096,
                out_dim=4096,
                dtype=DType.float8_e4m3fn,
                device=DeviceRef.GPU(),
                float8_config=config,
            )

        def __call__(self, x):
            return self.q_proj(x)

    m = TestModule()
    assert "q_proj" in m.sublayers

    # Check that raw_state_dict finds the weights with hierarchical names.
    state = m.raw_state_dict()
    assert "q_proj.weight" in state
    assert "q_proj.weight_scale" in state
    assert "q_proj.input_scale" in state


# ---- Sharding tests ----


def test_fat_tensor_shard_no_strategy_error() -> None:
    """Sharding without strategy raises error."""
    config = _make_fp8_config()
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.GPU(),
        float8_config=config,
    )
    with pytest.raises(ValueError, match="no sharding strategy"):
        ft.shard([DeviceRef.GPU(0)])


def test_fat_tensor_sharding_strategy_property() -> None:
    """Sharding strategy getter/setter works."""
    config = _make_fp8_config()
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.GPU(),
        float8_config=config,
    )
    assert ft.sharding_strategy is None

    strategy = ShardingStrategy.rowwise(num_devices=2)
    ft.sharding_strategy = strategy
    assert ft.sharding_strategy == strategy
    assert ft.weight.sharding_strategy == strategy


def test_fat_tensor_shard_rowwise_tensor_scale() -> None:
    """Test sharding with rowwise strategy and tensor-wise scale."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        config = _make_fp8_config(input_origin=Float8ScaleOrigin.STATIC)
        ft = FatTensor(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float8_e4m3fn,
            device=DeviceRef.GPU(0),
            float8_config=config,
        )
        ft.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = ft.shard(devices)
        assert len(shards) == 2

        for i, shard in enumerate(shards):
            # Weight is rowwise sharded.
            assert tuple(int(d) for d in shard.weight.shape) == (512, 4096)
            # Scalar weight_scale is shared (not sharded).
            assert len(shard.weight_scale.shape) == 0
            # Input scale is shared.
            assert shard.input_scale is ft.input_scale


def test_fat_tensor_shard_rowwise_rowwise_scale() -> None:
    """Test sharding with rowwise strategy and row-wise scale."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        config = _make_fp8_config(
            weight_granularity=Float8ScaleGranularity.ROWWISE
        )
        ft = FatTensor(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float8_e4m3fn,
            device=DeviceRef.GPU(0),
            float8_config=config,
        )
        ft.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = ft.shard(devices)

        for shard in shards:
            assert tuple(int(d) for d in shard.weight.shape) == (512, 4096)
            # Row-wise scale is sharded along with the weight.
            assert tuple(int(d) for d in shard.weight_scale.shape) == (512, 1)


def test_fat_tensor_shard_rowwise_block_scale() -> None:
    """Test sharding with rowwise strategy and block-wise scale."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        config = _make_fp8_config(
            weight_granularity=Float8ScaleGranularity.BLOCK,
            input_granularity=Float8ScaleGranularity.BLOCK,
            block_size=(128, 128),
        )
        ft = FatTensor(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float8_e4m3fn,
            device=DeviceRef.GPU(0),
            float8_config=config,
        )
        ft.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
        shards = ft.shard(devices)

        for shard in shards:
            assert tuple(int(d) for d in shard.weight.shape) == (512, 4096)
            # Block scale N-dim is sharded: 8/2 = 4, K-dim stays 32.
            assert tuple(int(d) for d in shard.weight_scale.shape) == (4, 32)


def test_fat_tensor_shard_colwise_rowwise_scale_replicated() -> None:
    """Rowwise scale is replicated for columnwise weight sharding."""
    config = _make_fp8_config(
        weight_granularity=Float8ScaleGranularity.ROWWISE
    )
    ft = FatTensor(
        in_dim=4096,
        out_dim=1024,
        dtype=DType.float8_e4m3fn,
        device=DeviceRef.CPU(),
        float8_config=config,
    )
    ft.sharding_strategy = ShardingStrategy.columnwise(num_devices=4)

    assert ft.weight_scale.sharding_strategy is not None
    assert ft.weight_scale.sharding_strategy.is_replicate


def test_fat_tensor_shard_preserves_config() -> None:
    """Sharding preserves float8_config on each shard."""
    with Graph(
        "test",
        input_types=[
            TensorType(DType.float32, (1, 4096), device=DeviceRef.GPU(0))
        ],
    ):
        config = _make_fp8_config(input_origin=Float8ScaleOrigin.STATIC)
        ft = FatTensor(
            in_dim=4096,
            out_dim=1024,
            dtype=DType.float8_e4m3fn,
            device=DeviceRef.GPU(0),
            float8_config=config,
        )
        ft.sharding_strategy = ShardingStrategy.rowwise(num_devices=2)

        shards = ft.shard([DeviceRef.GPU(0), DeviceRef.GPU(1)])
        for shard in shards:
            assert shard.float8_config is config
            assert shard.is_fp8
