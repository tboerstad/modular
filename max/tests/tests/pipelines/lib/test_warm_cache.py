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

"""Unit tests for the multi-model warm cache module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from max.pipelines.lib.warm_cache import (
    ModelInfo,
    _build_model_infos,
    _get_kernel_categories,
    _log_shared_kernel_info,
    _order_for_max_sharing,
    _DIFFUSION_KERNELS,
    _TRANSFORMER_DECODER_KERNELS,
    _TRANSFORMER_ENCODER_KERNELS,
)


def _make_mock_config(model_path: str) -> MagicMock:
    """Create a mock PipelineConfig with the given model path."""
    config = MagicMock()
    config.model.model_path = model_path
    config.model.huggingface_model_repo = MagicMock()
    config.use_legacy_module = True
    return config


def _make_mock_arch(name: str) -> MagicMock:
    """Create a mock SupportedArchitecture with the given name."""
    arch = MagicMock()
    arch.name = name
    return arch


def _make_model_info(
    model_path: str, arch_name: str, kernel_categories: frozenset[str]
) -> ModelInfo:
    """Create a ModelInfo for testing."""
    return ModelInfo(
        model_path=model_path,
        config=_make_mock_config(model_path),
        arch_name=arch_name,
        kernel_categories=kernel_categories,
    )


# ---------------------------------------------------------------------------
# Kernel category classification
# ---------------------------------------------------------------------------


class TestGetKernelCategories:
    """Tests for _get_kernel_categories."""

    def test_decoder_model(self) -> None:
        """Causal LM architectures get decoder kernels."""
        assert (
            _get_kernel_categories("LlamaForCausalLM") == _TRANSFORMER_DECODER_KERNELS
        )

    def test_gemma_is_decoder(self) -> None:
        """Gemma is a decoder architecture."""
        assert (
            _get_kernel_categories("GemmaForCausalLM") == _TRANSFORMER_DECODER_KERNELS
        )

    def test_encoder_model(self) -> None:
        """BERT-style architectures get encoder kernels."""
        assert _get_kernel_categories("BertModel") == _TRANSFORMER_ENCODER_KERNELS

    def test_mpnet_is_encoder(self) -> None:
        """MPNet is an encoder architecture."""
        assert _get_kernel_categories("MPNetModel") == _TRANSFORMER_ENCODER_KERNELS

    def test_diffusion_model(self) -> None:
        """Flux-style architectures get diffusion kernels."""
        assert _get_kernel_categories("FluxTransformer2DModel") == _DIFFUSION_KERNELS

    def test_unknown_returns_empty(self) -> None:
        """Unknown or None architecture returns empty set."""
        assert _get_kernel_categories(None) == frozenset()
        assert _get_kernel_categories("unknown") == frozenset()

    def test_cross_arch_kernel_overlap(self) -> None:
        """Decoder and encoder architectures share kernel types."""
        shared = _TRANSFORMER_DECODER_KERNELS & _TRANSFORMER_ENCODER_KERNELS
        assert "matmul" in shared
        assert "attention" in shared
        assert "normalization" in shared
        assert "activation" in shared
        assert "embedding" in shared
        assert "softmax" in shared

    def test_decoder_encoder_not_identical(self) -> None:
        """Decoder and encoder have different kernel sets (not a superset)."""
        assert _TRANSFORMER_DECODER_KERNELS != _TRANSFORMER_ENCODER_KERNELS
        # Decoder has kv_cache, rope, sampling; encoder has pooling.
        assert "kv_cache" in _TRANSFORMER_DECODER_KERNELS
        assert "kv_cache" not in _TRANSFORMER_ENCODER_KERNELS
        assert "pooling" in _TRANSFORMER_ENCODER_KERNELS
        assert "pooling" not in _TRANSFORMER_DECODER_KERNELS

    def test_diffusion_overlaps_with_decoder(self) -> None:
        """Diffusion models share matmul, normalization, attention with decoders."""
        shared = _DIFFUSION_KERNELS & _TRANSFORMER_DECODER_KERNELS
        assert "matmul" in shared
        assert "normalization" in shared
        assert "attention" in shared


# ---------------------------------------------------------------------------
# ModelInfo building
# ---------------------------------------------------------------------------


class TestBuildModelInfos:
    """Tests for _build_model_infos."""

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_single_model(self, mock_registry: MagicMock) -> None:
        """A single model produces one ModelInfo with correct kernel categories."""
        mock_registry.retrieve_architecture.return_value = _make_mock_arch(
            "LlamaForCausalLM"
        )
        config = _make_mock_config("model-a")

        infos = _build_model_infos([("model-a", config)])

        assert len(infos) == 1
        assert infos[0].arch_name == "LlamaForCausalLM"
        assert infos[0].kernel_categories == _TRANSFORMER_DECODER_KERNELS

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_mixed_architectures(self, mock_registry: MagicMock) -> None:
        """Models with different architectures get different kernel categories."""
        call_count = 0

        def side_effect(huggingface_repo, use_legacy_module):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_mock_arch("LlamaForCausalLM")
            return _make_mock_arch("BertModel")

        mock_registry.retrieve_architecture.side_effect = side_effect
        config_a = _make_mock_config("llama-model")
        config_b = _make_mock_config("bert-model")

        infos = _build_model_infos(
            [("llama-model", config_a), ("bert-model", config_b)]
        )

        assert len(infos) == 2
        assert infos[0].kernel_categories == _TRANSFORMER_DECODER_KERNELS
        assert infos[1].kernel_categories == _TRANSFORMER_ENCODER_KERNELS

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_unknown_architecture(self, mock_registry: MagicMock) -> None:
        """Unknown architecture gets empty kernel categories."""
        mock_registry.retrieve_architecture.return_value = None
        config = _make_mock_config("unknown-model")

        infos = _build_model_infos([("unknown-model", config)])

        assert infos[0].arch_name == "unknown"
        assert infos[0].kernel_categories == frozenset()


# ---------------------------------------------------------------------------
# Compilation ordering
# ---------------------------------------------------------------------------


class TestOrderForMaxSharing:
    """Tests for _order_for_max_sharing."""

    def test_models_with_more_kernels_first(self) -> None:
        """Models with more kernel categories should come first."""
        big = _make_model_info("big", "LlamaForCausalLM", _TRANSFORMER_DECODER_KERNELS)
        small = _make_model_info("small", "BertModel", _TRANSFORMER_ENCODER_KERNELS)

        # Decoder kernels have more categories than encoder.
        assert len(_TRANSFORMER_DECODER_KERNELS) > len(_TRANSFORMER_ENCODER_KERNELS)

        ordered = _order_for_max_sharing([small, big])
        assert ordered[0].model_path == "big"
        assert ordered[1].model_path == "small"

    def test_same_arch_grouped_together(self) -> None:
        """Models with the same architecture are grouped together."""
        llama_a = _make_model_info(
            "llama-a", "LlamaForCausalLM", _TRANSFORMER_DECODER_KERNELS
        )
        bert = _make_model_info("bert", "BertModel", _TRANSFORMER_ENCODER_KERNELS)
        llama_b = _make_model_info(
            "llama-b", "LlamaForCausalLM", _TRANSFORMER_DECODER_KERNELS
        )

        ordered = _order_for_max_sharing([bert, llama_a, llama_b])

        # Both Llama models should be adjacent (grouped by arch_name).
        llama_indices = [i for i, m in enumerate(ordered) if "llama" in m.model_path]
        assert llama_indices == [0, 1] or llama_indices == [1, 2]

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert _order_for_max_sharing([]) == []


# ---------------------------------------------------------------------------
# Shared kernel logging
# ---------------------------------------------------------------------------


class TestLogSharedKernelInfo:
    """Tests for _log_shared_kernel_info."""

    def test_same_arch_shared_kernels(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same-arch models are reported as sharing kernels."""
        infos = [
            _make_model_info(
                "model-a", "LlamaForCausalLM", _TRANSFORMER_DECODER_KERNELS
            ),
            _make_model_info(
                "model-b", "LlamaForCausalLM", _TRANSFORMER_DECODER_KERNELS
            ),
        ]

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_shared_kernel_info(infos)

        log_text = caplog.text
        assert "2 model(s)" in log_text
        assert "1 architecture(s)" in log_text
        assert "shared kernels" in log_text

    def test_cross_arch_reports_overlap(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Different architectures report cross-architecture kernel sharing."""
        infos = [
            _make_model_info(
                "llama-model", "LlamaForCausalLM", _TRANSFORMER_DECODER_KERNELS
            ),
            _make_model_info(
                "bert-model", "BertModel", _TRANSFORMER_ENCODER_KERNELS
            ),
        ]

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_shared_kernel_info(infos)

        log_text = caplog.text
        assert "2 model(s)" in log_text
        assert "2 architecture(s)" in log_text
        assert "Cross-architecture shared kernels" in log_text
        assert "LlamaForCausalLM <-> BertModel" in log_text
        # Should mention specific shared kernel types.
        assert "matmul" in log_text
        assert "attention" in log_text
        assert "overlap" in log_text.lower()

    def test_cross_arch_reports_common_across_all(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When all architectures share kernels, that is reported."""
        infos = [
            _make_model_info(
                "llama", "LlamaForCausalLM", _TRANSFORMER_DECODER_KERNELS
            ),
            _make_model_info(
                "bert", "BertModel", _TRANSFORMER_ENCODER_KERNELS
            ),
            _make_model_info(
                "flux", "FluxTransformer2DModel", _DIFFUSION_KERNELS
            ),
        ]

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_shared_kernel_info(infos)

        log_text = caplog.text
        assert "3 model(s)" in log_text
        assert "3 architecture(s)" in log_text
        assert "All architectures share" in log_text
        # matmul, normalization, activation, attention, embedding are common.
        assert "matmul" in log_text

    def test_single_arch_no_cross_arch_section(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A single architecture should not produce a cross-arch section."""
        infos = [
            _make_model_info(
                "model-a", "LlamaForCausalLM", _TRANSFORMER_DECODER_KERNELS
            ),
        ]

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_shared_kernel_info(infos)

        log_text = caplog.text
        assert "Cross-architecture" not in log_text
