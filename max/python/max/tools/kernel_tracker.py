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

"""Kernel tracking tool for MO/MOGG graphs.

Walks a MAX Graph's MLIR IR and extracts every operation (potential kernel)
along with its input/output shapes and dtypes.  Two modes:

  - **mo** (default): Walks the MO-dialect graph as built by Python.
    Shows which high-level ops the model uses and their shapes (including
    symbolic/parametric dimensions).

  - **mogg**: Runs the ``MOToMOGG`` compiler pass first, then walks the
    resulting MOGG-dialect IR.  Shows which concrete kernels were selected.

Usage as a library::

    from max.graph import Graph, TensorType, ops
    from max.tools.kernel_tracker import KernelTracker

    g = Graph("my_model")
    x = g.input(TensorType(DType.float32, [128, 784]))
    ...
    g.output(out)

    tracker = KernelTracker()
    report = tracker.track(g)
    print(json.dumps(report, indent=2))
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from max import _core
from max._core.dialects import builtin, kgen, mo, mosh
from max.graph import Graph

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_TENSOR_RE = re.compile(
    r"!mo\.(tensor|buffer)<\[([^\]]*)\],\s*(\w+)"
)


@dataclass
class TensorInfo:
    """Shape and dtype of a single tensor operand or result."""

    shape: list[str | int]
    dtype: str
    device: str = ""

    def is_symbolic(self) -> bool:
        return any(isinstance(d, str) for d in self.shape)


@dataclass
class OpRecord:
    """One operation/kernel hit in the graph."""

    position: int
    op_name: str
    op_class: str
    inputs: list[TensorInfo | None]
    outputs: list[TensorInfo | None]


@dataclass
class TrackingReport:
    """Full tracking report for a graph."""

    model: str
    level: str  # "mo" or "mogg"
    ops: list[OpRecord] = field(default_factory=list)
    symbolic_params: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["summary"] = self._summary()
        return d

    def _summary(self) -> dict[str, Any]:
        op_counts: Counter[str] = Counter()
        for op in self.ops:
            op_counts[op.op_name] += 1
        return {
            "total_ops": len(self.ops),
            "unique_ops": len(op_counts),
            "op_histogram": dict(op_counts.most_common()),
            "symbolic_params": self.symbolic_params,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# Core walker
# ---------------------------------------------------------------------------

# Structural ops that don't map to any kernel.
_SKIP_OP_CLASSES = {
    "ChainCreateOp",
    "ParamDeclareOp",
    "ParamFromValueOp",
    "ParamConstantOp",
    "OutputOp",
}


class KernelTracker:
    """Walk a MAX ``Graph`` and record every op with its shapes."""

    def track(
        self,
        graph: Graph,
        *,
        level: str = "mo",
    ) -> TrackingReport:
        """Produce a :class:`TrackingReport` for *graph*.

        Args:
            graph: A finalized MAX Graph.
            level: ``"mo"`` to walk the MO dialect (default), or ``"mogg"``
                to first lower through ``MOToMOGG`` and walk the result.

        Returns:
            A ``TrackingReport`` containing every op/kernel hit.
        """
        module = self._get_module(graph)

        if level == "mogg":
            self._lower_to_mogg(module)

        ops: list[OpRecord] = []
        symbolic: set[str] = set()
        position = 0

        for op in self._walk_ops(module):
            op_class = type(op).__name__
            if op_class in _SKIP_OP_CLASSES:
                continue

            record = self._extract_op(op, position)
            ops.append(record)
            position += 1

            # Collect symbolic dim names.
            for tensor in (*record.inputs, *record.outputs):
                if tensor is not None:
                    for dim in tensor.shape:
                        if isinstance(dim, str):
                            symbolic.add(dim)

        return TrackingReport(
            model=graph.name,
            level=level,
            ops=ops,
            symbolic_params=sorted(symbolic),
        )

    # ------------------------------------------------------------------
    # MLIR walking helpers (mirrors _interpreter.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_module(graph: Graph) -> builtin.ModuleOp:
        return _core.Operation._from_cmlir(graph._module.operation)  # type: ignore[return-value]

    @staticmethod
    def _lower_to_mogg(module: builtin.ModuleOp) -> None:
        """Apply MOToMOGG kernel-selection pass in-place."""
        ok = _core.lower(module, [mo.passes.MOToMOGG()])
        if not ok:
            raise RuntimeError("MOToMOGG lowering failed")

    @staticmethod
    def _walk_ops(module: builtin.ModuleOp) -> Iterator[_core.Operation]:
        """Yield every op inside every ``mo.graph`` in the module."""
        for top_level_op in module.body:
            if isinstance(top_level_op, mo.GraphOp):
                block = top_level_op.regions[0].front
                for op in block:
                    yield op

    # ------------------------------------------------------------------
    # Per-op extraction
    # ------------------------------------------------------------------

    def _extract_op(self, op: _core.Operation, position: int) -> OpRecord:
        op_class = type(op).__name__
        # Derive a friendly name: "MatmulOp" -> "mo.matmul"
        op_name = self._friendly_name(op_class, op)

        inputs: list[TensorInfo | None] = []
        for operand in op.operands:
            inputs.append(self._tensor_info(operand.value.type))

        outputs: list[TensorInfo | None] = []
        for result in op.results:
            outputs.append(self._tensor_info(result.type))

        return OpRecord(
            position=position,
            op_name=op_name,
            op_class=op_class,
            inputs=inputs,
            outputs=outputs,
        )

    # ------------------------------------------------------------------
    # Type parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_info(mlir_type: _core.Type) -> TensorInfo | None:
        """Extract shape/dtype from an MLIR type, or None for non-tensors."""
        # Try the typed API first.
        if isinstance(mlir_type, (mo.TensorType, mo.BufferType)):
            dtype = str(mlir_type.dtype).rsplit(".", 1)[-1]  # DType.float32 -> float32
            shape = _parse_shape_attr(mlir_type.shape_attr)
            device = ""
            try:
                device = mlir_type.device_ref.asm
            except Exception:
                pass
            return TensorInfo(shape=shape, dtype=dtype, device=device)

        # Fall back to parsing the textual representation.
        asm = mlir_type.asm
        m = _TENSOR_RE.search(asm)
        if m:
            shape = _parse_shape_str(m.group(2))
            return TensorInfo(shape=shape, dtype=m.group(3))

        # Non-tensor type (e.g. !mo.chain) – skip.
        return None

    @staticmethod
    def _friendly_name(op_class: str, op: _core.Operation) -> str:
        """Turn ``"MatmulOp"`` into ``"mo.matmul"`` etc."""
        # Strip trailing "Op"
        base = op_class.removesuffix("Op")
        # CamelCase -> snake_case
        name = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", base).lower()
        # Try to detect the dialect from the asm (first token after '=')
        asm = op.asm
        for prefix in ("mogg.", "mo.", "mosh."):
            if prefix in asm:
                # Get the full op name from asm, e.g. "mo.matmul"
                idx = asm.index(prefix)
                end = asm.index("(", idx) if "(" in asm[idx:] else len(asm)
                return asm[idx:end].strip()
        return f"mo.{name}"


# ---------------------------------------------------------------------------
# Shape parsing utilities
# ---------------------------------------------------------------------------


def _parse_shape_attr(attr: _core.Attribute) -> list[str | int]:
    """Parse a ShapeAttr into a list of int (static) or str (symbolic) dims."""
    if isinstance(attr, mosh.ShapeAttr):
        dims: list[str | int] = []
        for dim_attr in attr.values:
            dims.append(_parse_dim_attr(dim_attr))
        return dims
    # Unranked or parameter reference – fall back to asm.
    return _parse_shape_str(attr.asm)


def _parse_dim_attr(attr: _core.Attribute) -> str | int:
    """Parse a single dimension attribute (IntegerAttr or param ref)."""
    asm = attr.asm
    # IntegerAttr looks like "42 : index" or just a bare integer.
    # Try to extract an integer.
    int_match = re.match(r"^\s*(-?\d+)", asm)
    if int_match:
        return int(int_match.group(1))
    # Param reference like "D0", "batch", "seq_len", "add(N, 2)" etc.
    # Clean up and return as string.
    return asm.strip()


def _parse_shape_str(shape_str: str) -> list[str | int]:
    """Parse a shape string like ``"batch, 784"`` into ``["batch", 784]``."""
    dims: list[str | int] = []
    for part in shape_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            dims.append(int(part))
        except ValueError:
            dims.append(part)
    return dims


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m max.tools.kernel_tracker [options]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Track kernel/op hits and shapes in a MAX Graph."
    )
    parser.add_argument(
        "--level",
        choices=["mo", "mogg"],
        default="mo",
        help="Walk MO ops (default) or lower to MOGG first.",
    )
    parser.add_argument(
        "--output", "-o",
        default="-",
        help="Output file (default: stdout).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with a built-in test graph for demonstration.",
    )
    args = parser.parse_args(argv)

    if args.test:
        graph = _build_test_graph()
    else:
        parser.error(
            "Provide --test for a demo graph, or use the library API "
            "to pass your own Graph."
        )

    tracker = KernelTracker()
    report = tracker.track(graph, level=args.level)
    output = report.to_json(indent=2)

    if args.output == "-":
        sys.stdout.write(output + "\n")
    else:
        with open(args.output, "w") as f:
            f.write(output + "\n")


def _build_test_graph() -> Graph:
    """Build a small MLP graph for testing the tracker."""
    import numpy as np
    from max.dtype import DType
    from max.graph import TensorType, ops

    g = Graph("test_mlp")
    # input: [batch, 784]
    x = g.input(TensorType(DType.float32, ["batch", 784]))

    # Layer 1: Linear(784, 256) + ReLU
    w1 = g.constant(np.ones((784, 256), dtype=np.float32))
    b1 = g.constant(np.zeros(256, dtype=np.float32))
    h = ops.matmul(x, w1)
    h = ops.add(h, b1)
    h = ops.relu(h)

    # Layer 2: Linear(256, 10) + Softmax
    w2 = g.constant(np.ones((256, 10), dtype=np.float32))
    b2 = g.constant(np.zeros(10, dtype=np.float32))
    out = ops.matmul(h, w2)
    out = ops.add(out, b2)
    out = ops.softmax(out)

    g.output(out)
    return g


if __name__ == "__main__":
    main()
