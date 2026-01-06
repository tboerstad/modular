# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.abc import Sequence

from max.graph import TensorValue

from .layer import Module


class LayerList(Module):
    """Stores a list of modules.

    Can be used as a regular python list."""

    def __init__(self, layers: Sequence[Module]) -> None:
        super().__init__()
        self.layers = list(layers)

        for n, layer in enumerate(layers):
            self._sublayers[str(n)] = layer

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i: int) -> Module:
        return self.layers[i]

    def __delitem__(self, i: int) -> None:
        del self.layers[i]

    def __setitem__(self, i: int, layer: Module) -> None:
        self.layers[i] = layer

    def insert(self, i: int, layer: Module) -> None:
        self.layers.insert(i, layer)

    def append(self, layer: Module) -> None:
        self.layers.append(layer)

    def extend(self, layer: Module) -> None:
        self.layers.append(layer)

    def __str__(self) -> str:
        return str(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __call__(self, *args, **kwargs) -> TensorValue:
        x = self.layers[0](*args, **kwargs)
        for layer in self.layers[1:]:
            x = layer(x)
        return x
