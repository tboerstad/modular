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
    _group_by_architecture,
    _log_shared_kernel_info,
)


def _make_mock_config(model_path: str, arch_name: str) -> MagicMock:
    """Create a mock PipelineConfig with the given model path and arch name."""
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


class TestGroupByArchitecture:
    """Tests for _group_by_architecture."""

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_single_model(self, mock_registry: MagicMock) -> None:
        """A single model should produce a single group."""
        mock_registry.retrieve_architecture.return_value = _make_mock_arch(
            "LlamaForCausalLM_Legacy"
        )
        config = _make_mock_config("model-a", "LlamaForCausalLM_Legacy")

        groups = _group_by_architecture([("model-a", config)])

        assert len(groups) == 1
        assert "LlamaForCausalLM_Legacy" in groups
        assert len(groups["LlamaForCausalLM_Legacy"]) == 1

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_two_models_same_arch(self, mock_registry: MagicMock) -> None:
        """Two models with the same architecture should be grouped together."""
        mock_registry.retrieve_architecture.return_value = _make_mock_arch(
            "LlamaForCausalLM_Legacy"
        )
        config_a = _make_mock_config("model-a", "LlamaForCausalLM_Legacy")
        config_b = _make_mock_config("model-b", "LlamaForCausalLM_Legacy")

        groups = _group_by_architecture(
            [("model-a", config_a), ("model-b", config_b)]
        )

        assert len(groups) == 1
        assert len(groups["LlamaForCausalLM_Legacy"]) == 2

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_two_models_different_arch(self, mock_registry: MagicMock) -> None:
        """Two models with different architectures should produce two groups."""

        def side_effect(huggingface_repo, use_legacy_module):
            # Return different architectures based on call order
            if mock_registry.retrieve_architecture.call_count <= 1:
                return _make_mock_arch("LlamaForCausalLM_Legacy")
            return _make_mock_arch("BertModel_Legacy")

        mock_registry.retrieve_architecture.side_effect = side_effect
        config_a = _make_mock_config("llama-model", "LlamaForCausalLM_Legacy")
        config_b = _make_mock_config("bert-model", "BertModel_Legacy")

        groups = _group_by_architecture(
            [("llama-model", config_a), ("bert-model", config_b)]
        )

        assert len(groups) == 2
        assert "LlamaForCausalLM_Legacy" in groups
        assert "BertModel_Legacy" in groups

    @patch("max.pipelines.lib.warm_cache.PIPELINE_REGISTRY")
    def test_unknown_architecture(self, mock_registry: MagicMock) -> None:
        """Models with unknown architecture should be grouped under 'unknown'."""
        mock_registry.retrieve_architecture.return_value = None
        config = _make_mock_config("unknown-model", "unknown")

        groups = _group_by_architecture([("unknown-model", config)])

        assert "unknown" in groups


class TestLogSharedKernelInfo:
    """Tests for _log_shared_kernel_info."""

    def test_shared_kernels_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify shared kernel info is logged for models sharing an arch."""
        config_a = _make_mock_config("model-a", "LlamaForCausalLM_Legacy")
        config_b = _make_mock_config("model-b", "LlamaForCausalLM_Legacy")

        arch_groups = {
            "LlamaForCausalLM_Legacy": [
                ("model-a", config_a),
                ("model-b", config_b),
            ]
        }

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_shared_kernel_info(arch_groups)

        log_text = caplog.text
        assert "2 model(s)" in log_text
        assert "1 architecture(s)" in log_text
        assert "shared kernels" in log_text

    def test_no_sharing_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify output when models have different architectures."""
        config_a = _make_mock_config("model-a", "LlamaForCausalLM_Legacy")
        config_b = _make_mock_config("model-b", "BertModel_Legacy")

        arch_groups = {
            "LlamaForCausalLM_Legacy": [("model-a", config_a)],
            "BertModel_Legacy": [("model-b", config_b)],
        }

        with caplog.at_level("INFO", logger="max.pipelines"):
            _log_shared_kernel_info(arch_groups)

        log_text = caplog.text
        assert "2 model(s)" in log_text
        assert "2 architecture(s)" in log_text
