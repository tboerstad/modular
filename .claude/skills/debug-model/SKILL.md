---
name: debug-model
description: >
  Guide for debugging MAX model accuracy against a PyTorch reference.
  Use when a MAX model produces incorrect outputs, fails logit verification,
  or regresses numerically. Walks through the available tooling, common root
  causes, and how to reason efficiently through the problem.
---

# Debug Model

MAX is the **framework under test**. PyTorch is the **reference**.
The goal is to find where MAX and torch first diverge.

## Before You Start: Ask About Input Modality

If the model is a VLM (vision-language model), **ask the user now**:

> Should we debug with a text-only prompt, or with an image input?

The two paths differ in which components are active. Text-only is simpler —
start there unless the bug is specifically in the vision pathway.

## Classify the Problem First

Two main failure modes, each with different likely causes:

**New model implementation**
The model is freshly written and has never been correct. Look first at:
- Config parsing — are HuggingFace fields being read correctly?
- Weight loading — are weights mapped to the right layer names and shapes?
- RoPE / positional embeddings — frequency base, scaling, and dimension handling are common sources of silent errors

**Regression**
The model used to pass and now fails after a code change. Look first at:
- Recent kernel changes — a compute kernel (attention, matmul, softmax) may have introduced a numerical error
- Operator dispatch — a new code path may have been selected that behaves differently
- Quantization or dtype handling changes

Knowing which class you're in tells you where to spend your first 10 minutes.

## The Toolbox

These tools live under `max/tests/integration/tools/` and `max/tests/integration/accuracy/`.
Don't rely on exact CLI signatures — those change. Know what each tool *does*.

### `debug_model`
Runs a single pipeline (MAX or torch) and dumps intermediate layer tensors to
an output directory. This is the primary instrument for fine-grained comparison.
By default it runs with only 1 hidden layer — keep it that way until you need more.
Supports text prompts and image inputs for multimodal models.

### `compare_tensors`
Loads a pair of tensor files — one from torch (`.pt`), one from MAX (`.max`) —
and computes numerical difference metrics: max absolute difference, max relative
difference, and optionally a pass/fail against tolerances. Use this after
running `debug_model` on both frameworks to see which layers diverge.

### `generate_llm_logits`
Runs the full model end-to-end and saves the final logits to a JSON file.
Use this for a coarse check: if logits match, you're done. If not, you have
a baseline divergence to explain before drilling deeper.

### `verify` / `verify_pipelines`
Compares two logit JSON files (one from MAX, one from torch) using multiple
metrics: element-wise tolerance, cosine distance, and KL divergence.
`verify_pipelines` is the full automated pipeline; `verify` is the comparison step alone.

### `bisect_smoke_test`
Binary-searches across the model's layers to find the first layer where MAX
and torch diverge. Useful once you know *that* there's a divergence but not *where*.
Saves time compared to checking every layer manually.

### `hf_config_overrides`
Applies temporary overrides to the HuggingFace model config without changing
model files. Use this to isolate variables: reduce the number of layers,
change head counts, override RoPE parameters, etc.

## Debugging Strategy

**Start as simple as possible.** One layer, one decode step, default prompt.
A bug that exists with 1 layer exists with 32 layers — and is much faster to iterate on.

**Coarse before fine.** Compare final logits first with `generate_llm_logits` + `verify`.
If logits match within tolerance, the model is correct. If not, move to intermediate tensors.

**Intermediate tensors narrow the layer.** Run `debug_model` on both frameworks
with output paths, then use `compare_tensors` to walk through layers.
The first layer where tensors diverge is your target.

**Bisect if layer count is large.** `bisect_smoke_test` automates the binary
search so you don't have to manually check each layer.

**For regressions: check git history.** If you know the last good commit,
`git bisect` on the kernel or operator code is often faster than any tool above.

**For new models: read the reference config carefully.** Subtle differences —
`rope_theta`, `rope_scaling` type, tied embeddings, normalization placement —
are the most common source of correctness bugs in new architecture implementations.
Compare the HuggingFace `config.json` against what the MAX pipeline is actually using.

## Reading the Results

A divergence that grows with layer depth suggests an **accumulating error** —
often positional embeddings or normalization. A divergence that appears sharply
at one layer suggests a **local implementation bug** in that layer's operator.

Large absolute differences with small relative differences can indicate a
scale issue. Large relative differences on near-zero values are often noise
and may not matter for final output quality — check the logits, not just
intermediate tensors, to decide if it's worth pursuing.

Cosine distance is often the most stable metric for comparing distributions.
KL divergence amplifies errors in the tail of the softmax. Use both.
