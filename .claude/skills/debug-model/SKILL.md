---
name: debug-model
description: >
  Guide for debugging a MAX model bring-up against a PyTorch reference.
  Use when a new MAX model implementation produces incorrect outputs.
  Assumes the model is already known to be wrong — the goal is to find
  where in the Python stack MAX and torch first diverge.
---

# Debug Model

MAX is the **framework under test**. PyTorch is the **reference**.
We are bringing up a new model, so we assume it is already producing
incorrect results. The goal is to find exactly where in the Python stack
MAX first diverges from torch.

The bug is most likely in the Python part of the stack: config parsing,
weight loading, or the Python-level neural network operators.

## Understand the Bring-up Context First

Before touching any tool, spend a few minutes understanding what kind of
bring-up this is. The answer changes where you should look first.

**Is this a genuinely new architecture?**
If the model introduces new operators or custom kernels that were written
specifically for it, those kernels are untested and should be on your suspect
list alongside the Python layer. A kernel bug and a Python bug can look
identical from the outside.

**Does it use a new or unfamiliar kernel variant?**
Some models reuse existing kernel families (attention, MLP, normalization)
but in a configuration that hasn't been exercised before — different head
dimensions, unusual group sizes, a non-standard tiling. The kernel itself
may be correct in its common paths but untested in this one.

**Does it mostly reuse well-tested building blocks?**
If the architecture is a close relative of something already working in MAX
(same attention style, same MLP, same norm placement), the kernels are
probably fine and the bug is almost certainly in the Python: config parsing,
weight mapping, or a subtle structural difference between this model and
its relatives.

Ask these questions — of the person who filed the bug, of the PR that added
the model, or of the model's architecture documentation — before running
anything. A few minutes of context gathering can save hours of debugging in
the wrong layer of the stack.

## Before You Start: Ask About Input Modality

If the model is a VLM (vision-language model), **ask the user now**:

> Should we debug with a text-only prompt, or with an image input?

The two paths activate different components. Text-only is simpler and
eliminates the vision encoder from the picture — start there unless
the reported bug is specifically in visual understanding.

## The Toolbox

These tools live under `max/tests/integration/tools/`.
Know what each tool does — don't rely on specific CLI arguments, those change.

### `debug_model`
Runs a pipeline (MAX or torch) and dumps intermediate layer tensors to an
output directory. This is the primary instrument. Run it once for torch,
once for MAX, then compare. By default it runs with only 1 hidden layer —
keep it that way while isolating the problem.

### `compare_tensors`
Loads a pair of tensor files — one from torch (`.pt`), one from MAX (`.max`) —
and computes numerical difference metrics: max absolute difference, max relative
difference, and optionally a pass/fail against tolerances.

### `hf_config_overrides`
Applies temporary overrides to the HuggingFace model config without touching
model files. Use this to reduce layers, override RoPE parameters, swap head
counts, or otherwise control variables while reproducing the bug.

## Debugging Strategy

We know the model is wrong. The question is *where*.

**Start with 1 layer and 1 decode step.** A bug that exists in a 1-layer
model exists in a 32-layer model — and iterates 10x faster.

**Run `debug_model` on both frameworks and direct output to disk.**
Then walk through the intermediate tensors with `compare_tensors`.
The first layer where tensors diverge is your target. Work forward from
the model inputs — embedding outputs, attention inputs/outputs, MLP
inputs/outputs — until you see the divergence appear.

**When you find the diverging layer, read the Python implementation.**
Compare it directly against the HuggingFace reference implementation.
Look for subtle differences that are easy to miss:
- Wrong axis in a reshape or transpose
- A missing or extra normalization step
- An operation applied in the wrong order

**Config and weight loading are common culprits for brand-new models.**
If tensors diverge from the very first layer, suspect:
- A HuggingFace config field being silently misread or defaulted
- Weights loaded under wrong names or with wrong shapes
- A tied-embedding or bias term that exists in torch but not MAX (or vice versa)

**RoPE is the most common source of silent errors in attention.**
Check: frequency base (`rope_theta`), scaling type and factor, which
dimensions RoPE is applied to, and whether the implementation matches
the model's variant (standard, linear, dynamic, Llama-3 style, etc.).
A wrong RoPE produces plausible-looking but incorrect outputs — it will
not crash, so it can go unnoticed until tensor comparison.

**Normalization placement matters.** Pre-norm vs. post-norm, whether the
final layer norm is applied, and whether layer norms share weights are all
architecture details that differ between model families and are easy to copy
incorrectly.

## Reading the Results

A divergence that grows with layer depth suggests an **accumulating error** —
positional embeddings and normalization are the first suspects.

A divergence that appears sharply at one layer suggests a **local bug** in
that layer's operator — a wrong formula, wrong axis, or missing term.

Large absolute differences with small relative differences often point to a
scale issue (wrong normalization or missing weight scaling). Prioritize layers
with large absolute differences — they are more likely to affect final outputs.
