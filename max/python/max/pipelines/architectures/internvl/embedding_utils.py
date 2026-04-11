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

"""Re-exports merge_multimodal_embeddings from the shared location.

The canonical implementation now lives in max.pipelines.lib.vision_utils.
This module re-exports for backwards compatibility.
"""

from max.pipelines.lib.vision_utils import merge_multimodal_embeddings

__all__ = ["merge_multimodal_embeddings"]
