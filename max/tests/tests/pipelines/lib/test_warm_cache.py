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
    _log_compilation_plan,
    _order_for_max_sharing,
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


def _make_model_info(model_path: str, arch_name: str) -> ModelInfo:
    """Create a ModelInfo for testing."""
    return ModelInfo(
        model_path=model_path,
        config=_make_mock_config(model_path),
        arch_name=arch_name,
    )


# ---------------------------------------------------------------------------
# ModelInfo building
# ---------------------------------------------------------------------------


class TestBuildModelInfos:
    """Tests for _build_model_infos."""

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_single_model(self, mock_registry: MagicMock) -> None:
        """A single model produces one ModelInfo with the correct arch name."""
        mock_registry.retrieve_architecture.return_value = _make_mock_arch(
            "LlamaForCausalLM"
        )
        config = _make_mock_config("model-a")

        infos = _build_model_infos([("model-a", config)])

        assert len(infos) == 1
        assert infos[0].arch_name == "LlamaForCausalLM"
        assert infos[0].model_path == "model-a"

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_mixed_architectures(self, mock_registry: MagicMock) -> None:
        """Models with different architectures get different arch names."""
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
        assert infos[0].arch_name == "LlamaForCausalLM"
        assert infos[1].arch_name == "BertModel"

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_unknown_architecture(self, mock_registry: MagicMock) -> None:
        """Unknown architecture returns 'unknown' arch name."""
        mock_registry.retrieve_architecture.return_value = None
        config = _make_mock_config("unknown-model")

        infos = _build_model_infos([("unknown-model", config)])

        assert infos[0].arch_name == "unknown"


# ---------------------------------------------------------------------------
# Compilation ordering
# ---------------------------------------------------------------------------


class TestOrderForMaxSharing:
    """Tests for _order_for_max_sharing."""

    def test_same_arch_grouped_together(self) -> None:
        """Models with the same architecture are grouped together."""
        llama_a = _make_model_info("llama-a", "LlamaForCausalLM")
        bert = _make_model_info("bert", "BertModel")
        llama_b = _make_model_info("llama-b", "LlamaForCausalLM")

        ordered = _order_for_max_sharing([bert, llama_a, llama_b])

        # Both Llama models should be adjacent (grouped by arch_name).
        llama_indices = [
            i for i, m in enumerate(ordered) if "llama" in m.model_path
        ]
        assert llama_indices == [0, 1] or llama_indices == [1, 2]

    def test_stable_sort_within_same_arch(self) -> None:
        """Models with the same architecture maintain relative order."""
        a = _make_model_info("model-a", "LlamaForCausalLM")
        b = _make_model_info("model-b", "LlamaForCausalLM")
        c = _make_model_info("model-c", "LlamaForCausalLM")

        ordered = _order_for_max_sharing([a, b, c])

        paths = [m.model_path for m in ordered]
        assert paths == ["model-a", "model-b", "model-c"]

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert _order_for_max_sharing([]) == []

    def test_different_archs_sorted_by_name(self) -> None:
        """Different architectures are sorted by name for determinism."""
        gemma = _make_model_info("gemma", "GemmaForCausalLM")
        bert = _make_model_info("bert", "BertModel")
        llama = _make_model_info("llama", "LlamaForCausalLM")

        ordered = _order_for_max_sharing([gemma, bert, llama])

        arch_names = [m.arch_name for m in ordered]
        assert arch_names == sorted(arch_names)


# ---------------------------------------------------------------------------
# Compilation plan logging
# ---------------------------------------------------------------------------


class TestLogCompilationPlan:
    """Tests for _log_compilation_plan."""

    def test_same_arch_shared_kernels(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same-arch models are reported as sharing kernels."""
        infos = [
            _make_model_info("model-a", "LlamaForCausalLM"),
            _make_model_info("model-b", "LlamaForCausalLM"),
        ]

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_compilation_plan(infos)

        log_text = caplog.text
        assert "2 model(s)" in log_text
        assert "1 architecture(s)" in log_text
        assert "shared kernels" in log_text

    def test_cross_arch_reports_compiler_cache(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Different architectures report cross-architecture sharing via compiler cache."""
        infos = [
            _make_model_info("llama-model", "LlamaForCausalLM"),
            _make_model_info("bert-model", "BertModel"),
        ]

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_compilation_plan(infos)

        log_text = caplog.text
        assert "2 model(s)" in log_text
        assert "2 architecture(s)" in log_text
        assert "Cross-architecture kernel sharing enabled" in log_text
        assert "compiler cache" in log_text

    def test_single_model_no_cross_arch_section(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A single architecture should not produce a cross-arch section."""
        infos = [
            _make_model_info("model-a", "LlamaForCausalLM"),
        ]

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_compilation_plan(infos)

        log_text = caplog.text
        assert "Cross-architecture" not in log_text

    def test_three_architectures(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Three different architectures are all listed."""
        infos = [
            _make_model_info("llama", "LlamaForCausalLM"),
            _make_model_info("bert", "BertModel"),
            _make_model_info("flux", "FluxTransformer2DModel"),
        ]

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_compilation_plan(infos)

        log_text = caplog.text
        assert "3 model(s)" in log_text
        assert "3 architecture(s)" in log_text
        assert "LlamaForCausalLM" in log_text
        assert "BertModel" in log_text
        assert "FluxTransformer2DModel" in log_text
