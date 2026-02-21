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

"""Olmo3 Attention Layer.

Olmo3 now uses :class:`AttentionWithRope` from ``common_layers`` with
``qk_norm_full_dim=True`` for full-dimension QK normalization.

This module re-exports the class for backward compatibility.
"""

from ...common_layers.attention import AttentionWithRope

# Backward-compat alias.
Olmo3Attention = AttentionWithRope

__all__ = ["AttentionWithRope", "Olmo3Attention"]
