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

"""Track which kernels a MAX Graph hits and with what shapes.

Runs the MOToMOGG compiler pass (kernel selection) on a Graph, then walks
the resulting IR to record every kernel dispatch with its tensor shapes.

Library usage::

    from max.tools.kernel_tracker import track
    report = track(my_graph)

CLI usage::

    python -m max.tools.kernel_tracker --test -o report.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from max import _core
from max._core.dialects import builtin, mo, mosh
from max.graph import Graph

# Ops that are graph infrastructure, not compute kernels.
_SKIP_OPS = frozenset(
    {
        "mo.chain.create",
        "mo.output",
        "mo.rebind",
        "mo.constant",
        "mo.mutable.load",
        "mo.mutable.store",
        "kgen.param.declare",
        "mosh.param.to_value",
        "mosh.param.from_value",
    }
)

# Pattern to extract dialect-qualified op name from MLIR asm.
# Matches "mo.matmul", "mogg.fused_linear_relu", "mosh.broadcast", etc.
# Handles both bare and quoted forms (e.g. "mo.matmul"(...) vs mo.matmul(...)).
_OP_NAME_RE = re.compile(r'"?(\w+\.\w+(?:\.\w+)*)"?\s*[(<]')


def track(graph: Graph) -> dict[str, Any]:
    """Run kernel selection on *graph* and return every kernel hit.

    Applies ``MOToMOGG`` to select concrete kernels, then walks the IR.
    If lowering fails, falls back to the pre-lowering MO ops and warns.

    Returns a dict ready for ``json.dumps``::

        {
            "model": "llama3",
            "timestamp": "...",
            "commit": "abc123",
            "kernels": [ {op, inputs, outputs}, ... ],
            "summary": { ... }
        }
    """
    module: builtin.ModuleOp = _core.Operation._from_cmlir(  # type: ignore[assignment]
        graph._module.operation
    )

    lowered = _core.lower(module, [mo.passes.MOToMOGG()])
    if not lowered:
        print(
            "warning: MOToMOGG lowering failed, reporting MO-level ops",
            file=sys.stderr,
        )

    kernels: list[dict[str, Any]] = []
    sym_params: set[str] = set()

    for top_op in module.body:
        if not isinstance(top_op, mo.GraphOp):
            continue
        for op in top_op.regions[0].front:
            name = _op_name(op)
            if name in _SKIP_OPS:
                continue

            ins = [_tensor(o.value.type) for o in op.operands]
            outs = [_tensor(r.type) for r in op.results]

            # Drop entries that are all-None (pure chain ops that slipped through).
            if all(t is None for t in ins + outs):
                continue

            kernels.append({"op": name, "inputs": ins, "outputs": outs})

            for t in ins + outs:
                if t is not None:
                    for d in t["shape"]:
                        if isinstance(d, str):
                            sym_params.add(d)

    hist: Counter[str] = Counter(k["op"] for k in kernels)
    return {
        "model": graph.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_sha(),
        "kernels": kernels,
        "symbolic_params": sorted(sym_params),
        "summary": {
            "total": len(kernels),
            "unique": len(hist),
            "histogram": dict(hist.most_common()),
        },
    }


# -- MLIR helpers ----------------------------------------------------------


def _op_name(op: _core.Operation) -> str:
    """Extract the dialect-qualified op name from an operation's asm."""
    m = _OP_NAME_RE.search(op.asm)
    return m.group(1) if m else type(op).__name__


def _tensor(mlir_type: _core.Type) -> dict[str, Any] | None:
    """Return ``{"shape": [...], "dtype": "..."}`` or None for non-tensors."""
    if not isinstance(mlir_type, (mo.TensorType, mo.BufferType)):
        return None
    dtype = str(mlir_type.dtype).rsplit(".", 1)[-1]
    return {"shape": _shape(mlir_type.shape_attr), "dtype": dtype}


def _shape(attr: _core.Attribute) -> list[int | str]:
    """Parse a shape attribute into concrete ints and symbolic strings."""
    if isinstance(attr, mosh.ShapeAttr):
        return [_dim(d) for d in attr.values]
    return [attr.asm.strip()]  # unranked / param-ref


def _dim(attr: _core.Attribute) -> int | str:
    """Single dimension: integer if static, string if symbolic."""
    asm = attr.asm
    m = re.match(r"\s*(-?\d+)", asm)
    return int(m.group(1)) if m else asm.strip()


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except FileNotFoundError:
        return "unknown"


# -- CLI --------------------------------------------------------------------


def _build_test_graph() -> Graph:
    """Small MLP for smoke-testing the tracker."""
    import numpy as np
    from max.dtype import DType
    from max.graph import TensorType, ops

    g = Graph("test_mlp")
    x = g.input(TensorType(DType.float32, ["batch", 784]))

    w1 = g.constant(np.ones((784, 256), dtype=np.float32))
    b1 = g.constant(np.zeros(256, dtype=np.float32))
    h = ops.matmul(x, w1)
    h = ops.add(h, b1)
    h = ops.relu(h)

    w2 = g.constant(np.ones((256, 10), dtype=np.float32))
    b2 = g.constant(np.zeros(10, dtype=np.float32))
    out = ops.matmul(h, w2)
    out = ops.add(out, b2)
    out = ops.softmax(out)

    g.output(out)
    return g


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Track kernel hits in a MAX Graph.")
    p.add_argument("--test", action="store_true", help="Use built-in test graph.")
    p.add_argument("--output", "-o", default="-", help="Output file (default: stdout).")
    args = p.parse_args(argv)

    if not args.test:
        p.error("Pass --test, or use track() as a library with your own Graph.")

    report = track(_build_test_graph())
    out = json.dumps(report, indent=2) + "\n"

    if args.output == "-":
        sys.stdout.write(out)
    else:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
