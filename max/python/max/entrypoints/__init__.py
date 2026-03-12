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

"""High-level entrypoints for MAX pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.entrypoints.llm import LLM
    from max.pipelines.lib import PipelineConfig

__all__ = ["LLM", "PipelineConfig"]


def __getattr__(name: str) -> object:
    if name == "LLM":
        from max.entrypoints.llm import LLM

        return LLM
    if name == "PipelineConfig":
        from max.pipelines.lib import PipelineConfig

        return PipelineConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
