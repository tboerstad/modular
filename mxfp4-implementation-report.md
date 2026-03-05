# MXFP4 Implementation in vLLM and SGLang: A Comprehensive Report

## Table of Contents

1. [Background: MX Microscaling Formats](#1-background-mx-microscaling-formats)
2. [MXFP4 vs NVFP4: Two Flavors of FP4](#2-mxfp4-vs-nvfp4-two-flavors-of-fp4)
3. [vLLM Implementation](#3-vllm-implementation)
4. [SGLang Implementation](#4-sglang-implementation)
5. [Comparative Analysis: Upcasting Strategies](#5-comparative-analysis-upcasting-strategies)
6. [Comparative Analysis: Kernel Dispatch](#6-comparative-analysis-kernel-dispatch)
7. [Key Takeaways](#7-key-takeaways)

---

## 1. Background: MX Microscaling Formats

The **OCP Microscaling (MX) v1.0 Specification** (September 2023) defines a family of block-scaled narrow-precision formats. Backed by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm, it specifies four formats: MXFP8, MXFP6, MXFP4, and MXINT8.

### MXFP4 Element Format: E2M1

- **4 bits total**: 1 sign bit, 2 exponent bits, 1 mantissa bit
- Exponent bias: 1
- No encodings for Inf or NaN
- **16 representable values**: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} and their negatives

### Block Structure

- **Block size**: 32 contiguous elements along the K (reduction) dimension
- **Scale format**: E8M0 (8-bit, exponent-only, zero mantissa bits, bias 127)
  - Represents only powers of 2, from 2^(-127) to 2^(127)
  - Value 0xFF reserved for NaN
- **Storage per block**: 32 x 4 bits + 8 bits = 136 bits (17 bytes)
- **Effective bits/element**: ~4.25

### How Block Scaling Works for GEMM

For `C = A * B`, each contiguous group of 32 elements along K in both A and B has its own E8M0 scale factor. The hardware or software multiplies each micro-block's elements by its scale before or during accumulation. Accumulation is typically done in FP32.

---

## 2. MXFP4 vs NVFP4: Two Flavors of FP4

Both vLLM and SGLang implement **two distinct FP4 systems** that share the same E2M1 data element but differ in scale format and block size:

| Property              | MXFP4 (OCP Standard)       | NVFP4 (NVIDIA Proprietary) |
|-----------------------|-----------------------------|----------------------------|
| Element format        | E2M1 (4-bit)               | E2M1 (4-bit)              |
| Block size            | **32** elements             | **16** elements            |
| Scale format          | **E8M0** (power-of-2 only) | **E4M3** (FP8, finer)     |
| Per-tensor scale      | No                          | **Yes** (FP32)             |
| Effective bits/element| ~4.25                       | ~4.5                       |
| Accuracy              | Lower (coarser scales)      | Higher (3 levels of scale) |
| Min GPU arch          | SM80 (Hopper emulation)     | SM100 (Blackwell native)   |

NVFP4's smaller block size (16 vs 32) provides twice as many opportunities to match local dynamic range. Using E4M3 (which can represent non-power-of-2 values) instead of E8M0 for scales further reduces quantization error. The additional per-tensor FP32 scale factor adds a third level of scaling.

---

## 3. vLLM Implementation

### 3.1 Quantization Config Classes

| Config Class                     | Registry Name         | Scope                              |
|----------------------------------|-----------------------|------------------------------------|
| `Mxfp4Config`                    | `"mxfp4"`            | MoE only (linear falls back to unquantized) |
| `ModelOptNvFp4Config`            | `"modelopt_fp4"`     | Linear + MoE + KV cache           |
| `FPQuantConfig`                  | `"fp_quant"`         | Linear only (mxfp4 or nvfp4)      |
| `CompressedTensorsConfig`        | `"compressed-tensors"` | Delegates to scheme classes      |
| `QuarkConfig`                    | `"quark"`            | Delegates to `QuarkOcpMxScheme`   |
| `PetitNvFp4Config`              | `"petit_nvfp4"`      | NVFP4 with Petit transforms       |

Key scheme classes under compressed-tensors:
- `CompressedTensorsW4A16Mxfp4` -- MXFP4 weight-only, uses Marlin
- `CompressedTensorsW4A4Fp4` -- NVFP4 W4A4, uses CUTLASS/Marlin/FlashInfer
- `CompressedTensorsW4A16Fp4` -- NVFP4 weight-only

**Key files:**
- `vllm/model_executor/layers/quantization/mxfp4.py`
- `vllm/model_executor/layers/quantization/modelopt.py`
- `vllm/model_executor/layers/quantization/fp_quant.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`

### 3.2 Weight Loading

**MXFP4 weights:**
- Data: E2M1 packed 2-per-byte into `torch.uint8`, shape `[N, K//2]` (linear) or `[E, N, K//2]` (MoE)
- Scales: E8M0 stored as `torch.uint8`, shape `[N, K//32]`
- No global scale

**NVFP4 weights:**
- Data: E2M1 packed 2-per-byte into `torch.uint8`, shape `[N, K//2]`
- Block scales: `torch.float8_e4m3fn`, shape `[N, K//16]`
- Global weight scale: `torch.float32` scalar (`weight_scale_2`)
- Input global scale: `torch.float32` scalar for activation quantization

### 3.3 Upcasting Strategy

#### NVFP4 (W4A4 path -- primary path on Blackwell)

**There is NO explicit weight upcasting.** Both activations and weights remain in FP4 format:

1. **Activations** are dynamically quantized from BF16/FP16 **down** to FP4 on GPU via `scaled_fp4_quant()` (a CUDA kernel in `csrc/quantization/fp4/nvfp4_quant_kernels.cu`). This computes per-block-of-16 max, derives E4M3 scale, and quantizes to E2M1.
2. **The CUTLASS GEMM kernel** consumes FP4 x FP4 natively on Blackwell tensor cores (SM100+), accumulates in FP32, and outputs BF16/FP16 directly.
3. The alpha scalar (`input_global_scale * weight_global_scale`) is applied in the GEMM epilogue.

**No upcasting happens -- the GEMM runs natively on FP4 data.**

#### NVFP4 on Pre-Blackwell (Marlin fallback)

For GPUs < SM100, the **Marlin GEMM kernel** dequantizes FP4 weights to FP16/BF16 **on-the-fly during GEMM execution** on GPU:
- `marlin_gemm()` with `b_q_type=scalar_types.float4_e2m1f`
- The dequantization is fused into the matrix multiply -- there is no separate dequant step
- Activations stay in FP16/BF16 (W4A16 mode)

#### MXFP4 (Weight-only W4A16 path)

- Activations stay in BF16
- On the **Marlin path**: E8M0 scales are converted to param dtype (FP16/BF16) for Marlin's built-in dequantization
- On the **Triton/FlashInfer path**: weights stay packed; the kernel dequantizes internally
- On the **emulation path**: `run_nvfp4_emulations()` dequantizes both weights and activations to **FP32** on GPU, then runs `torch.matmul` -- used only for testing

#### FPQuant path (Qutlass)

- `fusedQuantizeMx()` -- Hadamard rotation + MXFP4 quantization fused kernel
- `matmul_mxf4_bf16_tn()` -- MXFP4 x BF16 matmul via Qutlass
- Weights are in MXFP4, activations in BF16

#### ROCm/Quark (AMD)

- `dequant_mxfp4` calls `quark.torch.kernel.mx.dq_mxfp4()` to dequantize on GPU
- `aiter.gemm_a4w4` for ASM-based FP4 GEMM on AMD GPUs

### 3.4 Kernels Called

#### CUTLASS FP4 GEMM (SM100+, SM120+) -- Dense Linear

```
Python:   cutlass_scaled_fp4_mm()              [vllm/_custom_ops.py]
C++ entry: cutlass_scaled_fp4_mm()             [csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu]
SM100:    cutlass_scaled_fp4_mm_sm100a()       [csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu]
SM120:    cutlass_scaled_fp4_mm_sm120a()       [csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu]
```

These use **CUTLASS 3.x `GemmUniversalAdapter`** with:
- Element type: `cutlass::nv_float4_t<cutlass::float_e2m1_t>`
- Operator class: `OpClassBlockScaledTensorOp`
- Scale type: `cutlass::float_ue4m3_t` (unsigned E4M3)
- Tile shapes by M dimension:
  - M <= 16: 128x128x256
  - M <= 256: 256x128x256
  - M > 256: 256x256x256
- Accumulation: FP32
- Output: BF16 or FP16

#### Activation Quantization Kernels (CUDA)

```
scaled_fp4_quant()                             -- BF16/FP16 -> FP4 quantization
scaled_fp4_experts_quant()                     -- per-expert activation quant for MoE
silu_and_mul_nvfp4_quant()                     -- fused SiLU+Mul+FP4 quantization
silu_and_mul_scaled_fp4_experts_quant()        -- fused SiLU+Mul+FP4 for MoE experts
```

All in `csrc/quantization/fp4/nvfp4_quant_kernels.cu`.

#### MoE GEMM (CUTLASS Grouped)

```
cutlass_fp4_moe_mm()                           [csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu]
```

Orchestrated by `run_cutlass_moe_fp4()` in `vllm/model_executor/layers/fused_moe/cutlass_moe.py`:
1. `shuffle_rows` -- permute activations by expert assignment
2. `scaled_fp4_experts_quant` -- per-expert activation quantization
3. `cutlass_fp4_moe_mm` -- FP4 grouped GEMM for gate/up projection
4. `silu_and_mul_scaled_fp4_experts_quant` -- fused activation + FP4 quant
5. `cutlass_fp4_moe_mm` -- second grouped GEMM for down projection
6. Unpermute and weighted reduce

#### Marlin FP4 GEMM (fallback for pre-Blackwell)

```
marlin_gemm() with b_q_type=scalar_types.float4_e2m1f   [csrc/quantization/marlin/marlin.cu]
```

Works on any GPU >= SM75. Dequantizes FP4 to FP16/BF16 on-the-fly during GEMM.

#### FlashInfer Backends

```
flashinfer_scaled_fp4_mm()         -- CUTLASS, TRTLLM, or cuDNN backend
flashinfer.fused_moe               -- MoE with FP4 experts (TRTLLM, CUTLASS, CuteDSL)
```

#### Triton Kernels (MXFP4 MoE)

```
triton_kernels.matmul_ogs          -- with PrecisionConfig for MXFP4
```

### 3.5 Scale Factor Processing

**NVFP4 scale swizzling for CUTLASS** (`swizzle_blockscale()` in `utils/nvfp4_utils.py`):
- Pad rows to multiples of 128, columns to multiples of 4
- Reshape into `[M_padded//128, 4, 32, K_padded//4, 4]`
- Permute to match hardware's 128x4 scale factor tile layout
- This produces the memory layout expected by CUTLASS's `tcgen05.mma.blockscaled` instruction

**MXFP4 scale swizzling for Marlin** (`mxfp4_marlin_process_scales()`):
- Converts to `torch.float8_e8m0fnu`
- Optionally adjusts exponent bias (+6 for FP8 activation path)
- Permutes columns for Marlin's dequant layout

### 3.6 NVFP4 Dense Linear Forward Pass (`apply_nvfp4_linear()`)

```python
def apply_nvfp4_linear(x_bf16, weight_fp4, weight_scale_e4m3, alpha_f32, ...):
    # Step 1: Quantize activations to FP4 on GPU
    x_fp4, x_blockscale = scaled_fp4_quant(x_bf16, input_global_scale_inv)

    # Step 2: Pad if needed for CUTLASS alignment
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4)

    # Step 3: GEMM -- dispatches to one of:
    #   - cutlass_scaled_fp4_mm()          (SM100+, native FP4)
    #   - flashinfer_scaled_fp4_mm()       (CUTLASS/TRTLLM/cuDNN)
    #   - apply_fp4_marlin_linear()        (pre-Blackwell, dequant on-the-fly)
    #   - run_nvfp4_emulations()           (testing only, dequant to FP32)
    out = cutlass_scaled_fp4_mm(x_fp4, weight_fp4, x_blockscale, weight_scale_e4m3, alpha, out_dtype)

    # Step 4: Slice output, add bias
    return out[:, :N] + bias
```

---

## 4. SGLang Implementation

### 4.1 Quantization Config Classes

| Config Class              | Registry Name         | Scope                              |
|---------------------------|-----------------------|------------------------------------|
| `ModelOptFp4Config`       | `"modelopt_fp4"`     | Linear + MoE                       |
| `Mxfp4Config`            | `"mxfp4"`            | MoE (AMD ROCm + NVIDIA)           |
| `CompressedTensorsConfig` | `"compressed-tensors"` | Auto-detects W4A4 NVFP4         |
| `QuarkConfig`            | `"quark"`             | Dispatches to `QuarkW4A4MXFP4`   |
| `PetitNvFp4Config`       | `"petit_nvfp4"`      | NVFP4 with Petit transforms       |

**Key files:**
- `sglang/srt/layers/quantization/modelopt_quant.py`
- `sglang/srt/layers/quantization/mxfp4.py`
- `sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
- `sglang/srt/layers/quantization/fp4_utils.py`

### 4.2 Weight Loading

**NVFP4 (ModelOpt) -- 4 tensors per linear layer:**

| Tensor          | Dtype            | Shape              | Description                          |
|-----------------|------------------|--------------------|--------------------------------------|
| `weight`        | uint8 (packed)   | `[N, K/2]`        | Two E2M1 values per byte             |
| `weight_scale`  | float8_e4m3fn    | `[N, K/16]`       | Per-block E4M3 scale factors         |
| `weight_scale_2`| float32          | scalar             | Global weight scale                  |
| `input_scale`   | float32          | scalar             | Global activation scale              |

Alpha = `input_scale * weight_scale_2`.

**Compressed Tensors NVFP4** -- same format, different tensor naming:
- `weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale`
- Alpha = `1 / (input_global_scale * weight_global_scale)` -- **note the inverse convention**

**OCP MXFP4:**
- Weights: uint8 (packed E2M1), block size **32**
- Scales: uint8 (E8M0)
- No global scale

### 4.3 Upcasting Strategy

#### NVFP4 (Primary Path)

**No explicit upcasting of weights.** Same approach as vLLM:

1. **Activations** dynamically quantized from BF16/FP16 **down** to FP4 via `fp4_quantize(x, layer.input_scale_inv)` -- converts BF16/FP16 to E2M1 + swizzled E4M3 block scales
2. **CUTLASS GEMM** consumes FP4 x FP4 natively on Blackwell, accumulates in FP32, outputs BF16/FP16
3. Alpha applied in GEMM epilogue

**No weight upcasting -- native FP4 execution.**

#### OCP MXFP4 (Fallback Path)

The **only upcasting path** is a fallback in `mxfp4.py` (lines 720-739) when triton_kernels are not available:
```python
upcast_from_mxfp(weight, scale, target_dtype=torch.bfloat16)
```
This dequantizes weights to **BF16** at **model load time** (CPU-side), then runs standard BF16 GEMM at inference. This is the non-optimized fallback.

On the optimized paths:
- **AMD ROCm**: `dynamic_mxfp4_quant` from Quark/AIter for dynamic quantization, `aiter.fused_moe.fused_moe` with `QuantType.per_1x32` for the GEMM
- **NVIDIA with FlashInfer**: Weights stay packed, FlashInfer TRT-LLM kernel handles dequant internally

### 4.4 Kernels Called

#### JIT CUTLASS FP4 GEMM -- Dense Linear

```
Python wrapper:  cutlass_scaled_fp4_mm          [sglang/jit_kernel/nvfp4.py, line 220]
CUDA kernel:     nvfp4_scaled_mm_kernels.cuh    [sglang/jit_kernel/csrc/gemm/nvfp4/]
```

Uses **CUTLASS 3.x `GemmUniversal`** with:
- Element type: `cutlass::nv_float4_t<cutlass::float_e2m1_t>`
- Operator class: `cutlass::arch::OpClassBlockScaledTensorOp`
- SM100 schedules: `KernelTmaWarpSpecialized1SmNvf4Sm100` (M<=128) and `KernelTmaWarpSpecialized2SmNvf4Sm100` (M>128)
- SM120 support via `Fp4GemmSm120` template
- Tile shapes: 128x256x256 or 256x256x256 depending on M
- Output: BF16, FP16, or FP32

**Key difference from vLLM**: SGLang uses **JIT-compiled** CUTLASS kernels (compiled at first use), while vLLM pre-compiles them as part of the build.

#### Activation Quantization Kernel

```
CUDA kernel:     nvfp4_quant_kernels.cuh        [sglang/jit_kernel/csrc/gemm/nvfp4/]
Function:        cvt_fp16_to_fp4                 -- each thread converts 8 BF16/FP16 values to E2M1
```

Per-block max computation, E4M3 scale derivation: `SF = SFScaleVal * (vecMax / 6.0)`, conversion via `fp32_vec_to_e2m1()`. Scale output is blockwise-interleaved (swizzled) for efficient TMA access.

#### FlashInfer Backends

```
flashinfer.mm_fp4(input, weight, input_sf, weight_sf, alpha, out_dtype, backend=...)
```
With backends:
- `"cutlass"` -- CUTLASS-based
- `"trtllm"` -- TensorRT-LLM-based
- `"cudnn"` -- cuDNN-based

#### MoE GEMM Kernels

**CUTLASS grouped GEMM:**
```
cutlass_fp4_group_mm                             [sglang/jit_kernel/nvfp4.py, line 236]
CUDA kernel:     nvfp4_blockwise_moe.cuh         [sglang/jit_kernel/csrc/moe/]
```

**FlashInfer MoE:**
```
flashinfer.fused_moe.cutlass_fused_moe           -- CUTLASS fused MoE
flashinfer.trtllm_fp4_block_scale_moe            -- TRT-LLM MoE
```

**AMD ROCm:**
```
aiter.fused_moe.fused_moe with QuantType.per_1x32
```

#### MoE Orchestration (`cutlass_moe_fp4()`)

```python
def cutlass_moe_fp4(hidden_states, w1, w2, ...):
    # 1. Route tokens to experts
    prepare_moe_input(...)

    # 2. Quantize activations to FP4 per expert
    scaled_fp4_experts_quant(...)

    # 3. First grouped GEMM (up-projection)
    cutlass_fp4_group_mm(...)

    # 4. Activation function
    silu_and_mul(...)

    # 5. Quantize intermediate to FP4
    scaled_fp4_experts_quant(...)

    # 6. Second grouped GEMM (down-projection)
    cutlass_fp4_group_mm(...)

    # 7. Combine expert outputs
    apply_shuffle_mul_sum(...)
```

### 4.5 Scale Factor Processing

**NVFP4 swizzling** (`swizzle_blockscale()` in `utils.py`):
- Reshapes `[N, K/16]` to `[N/128, 4, 32, K/16/4, 4]`
- Permutes to `[N/128, K/16/4, 32, 4, 4]`
- Pads N to 128 and K/16 to 4
- Produces memory layout matching CUTLASS's TMA access pattern

**OCP MXFP4 on AMD**: `e8m0_shuffle()` rearranges scales for hardware consumption.

**OCP MXFP4 on NVIDIA**: `nvfp4_block_scale_interleave()` from FlashInfer rearranges for TMA.

### 4.6 FP4 GEMM Backend Selection

SGLang exposes backend selection via `--fp4-gemm-backend` server arg or `SGLANG_FLASHINFER_FP4_GEMM_BACKEND` env var:
- `auto` (default)
- `flashinfer_cutlass`
- `flashinfer_trtllm`
- `flashinfer_cudnn`

The `fp4_gemm()` dispatcher in `modelopt_quant.py` tries FlashInfer first, falling back to JIT CUTLASS.

### 4.7 NVFP4 Dense Linear Forward Pass

```python
# ModelOptFp4LinearMethod.apply(), line ~1276 of modelopt_quant.py

def apply(layer, x, bias):
    # Step 1: Quantize activations to FP4 on GPU
    x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)

    # Step 2: Pad x_fp4 if K was padded for alignment
    x_fp4 = pad_if_needed(x_fp4)

    # Step 3: FP4 GEMM (dispatches to CUTLASS or FlashInfer)
    out = fp4_gemm(x_fp4, layer.weight, x_scale, layer.weight_scale, layer.alpha, out_dtype, N)

    # Step 4: Slice output to remove N-dimension padding
    out = out[:, :N]

    # Step 5: Add bias if present
    return out + bias
```

---

## 5. Comparative Analysis: Upcasting Strategies

### When and Where Upcasting Happens

| Scenario                        | vLLM                                          | SGLang                                        |
|---------------------------------|-----------------------------------------------|-----------------------------------------------|
| **NVFP4 on Blackwell (SM100+)**| **No upcasting**. FP4 x FP4 GEMM natively on tensor cores. Accumulate in FP32, output BF16. | **No upcasting**. Same approach -- native FP4 GEMM via CUTLASS. Accumulate in FP32, output BF16. |
| **NVFP4 on pre-Blackwell**     | **Marlin dequant on-the-fly**: FP4 -> FP16/BF16 fused into GEMM. No separate dequant step. | Not explicitly supported (requires SM100+).    |
| **MXFP4 weight-only (Marlin)** | **Marlin dequant on-the-fly**: E2M1 -> FP16/BF16 fused into GEMM. E8M0 scales converted to param dtype. | N/A (MXFP4 linear falls back to BF16 GEMM via `upcast_from_mxfp` at load time). |
| **MXFP4 MoE (Triton)**         | Triton kernel dequants internally.            | Triton kernel dequants internally.            |
| **MXFP4 fallback**             | Emulation: dequant to **FP32** on GPU.        | Fallback: dequant to **BF16** at model load time (CPU-side). |
| **MXFP4 on AMD (ROCm)**        | `quark.dq_mxfp4()` on GPU.                   | `aiter.fused_moe` with `QuantType.per_1x32`. |

### What Format They Upcast To

| Path                            | Source Format    | Upcast Target    | Where             |
|---------------------------------|------------------|------------------|--------------------|
| Blackwell CUTLASS (both)        | E2M1 (FP4)      | **None** (native) | N/A -- tensor core |
| Marlin fallback (vLLM)          | E2M1 (FP4)      | **FP16 / BF16**  | GPU, fused in GEMM |
| MXFP4 fallback (SGLang)        | E2M1 (FP4)      | **BF16**          | CPU, at load time  |
| Emulation (vLLM)                | E2M1 (FP4)      | **FP32**          | GPU, at runtime    |
| FPQuant/Qutlass (vLLM)         | E2M1 (FP4)      | **BF16** (activations stay BF16) | GPU, mixed GEMM |
| Hopper CUTLASS emulation        | E2M1 (FP4)      | **FP32 -> TF32**  | GPU, CUTLASS software path |

### Key Insight

**On Blackwell (the target production hardware), neither framework upcasts at all.** The Blackwell tensor cores (`tcgen05.mma.blockscaled`) natively consume packed FP4 data with block scale factors and perform the multiply-accumulate in one fused operation. The FP4 -> higher precision conversion happens implicitly inside the tensor core hardware, never as an explicit software step.

The only explicit upcasting occurs in **fallback paths** for older GPUs:
- **Marlin** (vLLM): fuses FP4->FP16/BF16 dequant into the GEMM kernel
- **Load-time dequant** (SGLang MXFP4 fallback): converts to BF16 at model load, then runs standard BF16 GEMM
- **Emulation** (vLLM testing): converts to FP32 for reference/debugging
- **Hopper CUTLASS** (software path): converts E2M1 -> FP32, then uses TF32 tensor cores

---

## 6. Comparative Analysis: Kernel Dispatch

### Dense Linear GEMM

| Backend          | vLLM                                  | SGLang                                |
|------------------|---------------------------------------|---------------------------------------|
| CUTLASS (SM100)  | Pre-compiled `cutlass_scaled_fp4_mm_sm100a()` | **JIT-compiled** `cutlass_scaled_fp4_mm` |
| CUTLASS (SM120)  | Pre-compiled `cutlass_scaled_fp4_mm_sm120a()` | JIT-compiled SM120 variant            |
| FlashInfer       | `flashinfer_scaled_fp4_mm()`          | `flashinfer.mm_fp4()` with backend arg |
| Marlin (fallback)| `marlin_gemm()` with `float4_e2m1f`  | Not available for NVFP4               |
| FBGEMM           | `torch.ops.fbgemm.f4f4bf16()`        | Not available                         |
| Qutlass          | `matmul_mxf4_bf16_tn()` (FPQuant)    | Not available                         |

### MoE GEMM

| Backend           | vLLM                                 | SGLang                                |
|--------------------|--------------------------------------|---------------------------------------|
| CUTLASS grouped    | `cutlass_fp4_moe_mm()` pre-compiled  | `cutlass_fp4_group_mm()` JIT-compiled |
| FlashInfer TRTLLM  | `TrtLlmGenExperts`                   | `flashinfer.trtllm_fp4_block_scale_moe` |
| FlashInfer CUTLASS | `FlashInferExperts`                  | `flashinfer.fused_moe.cutlass_fused_moe` |
| Triton             | `OAITritonExperts` with MXFP4        | Not used for NVFP4                    |
| AMD/CK             | `aiter` CK MoE kernel               | `aiter.fused_moe.fused_moe`          |

### Activation Quantization

| Operation                    | vLLM                                     | SGLang                                    |
|------------------------------|------------------------------------------|-------------------------------------------|
| BF16 -> FP4                  | `scaled_fp4_quant()` (CUDA)              | `fp4_quantize()` -> JIT CUDA or FlashInfer |
| Per-expert quant             | `scaled_fp4_experts_quant()` (CUDA)      | `scaled_fp4_experts_quant()` (JIT CUDA)   |
| Fused SiLU+Mul+FP4          | `silu_and_mul_nvfp4_quant()` (CUDA)      | Separate SiLU then quant                  |

### Backend Selection

**vLLM** selects NVFP4 MoE backend via priority:
```
FLASHINFER_TRTLLM > FLASHINFER_CUTEDSL > FLASHINFER_CUTLASS > VLLM_CUTLASS > MARLIN
```

**SGLang** selects via `--fp4-gemm-backend` or env var:
```
auto | flashinfer_cutlass | flashinfer_trtllm | flashinfer_cudnn
```

---

## 7. Key Takeaways

### Architecture Summary

Both vLLM and SGLang follow a nearly identical high-level architecture for NVFP4:

```
BF16 activations
    |
    v
[Activation quantization kernel]  -- BF16 -> E2M1 (FP4) + E4M3 block scales
    |
    v
[CUTLASS BlockScaledTensorOp GEMM]  -- FP4 x FP4, accumulate FP32, output BF16
    |                                   nv_float4_t<float_e2m1_t> element type
    |                                   OpClassBlockScaledTensorOp operator
    v
BF16 output (with alpha epilogue scaling)
```

### Key Differences

1. **JIT vs Pre-compiled kernels**: SGLang JIT-compiles CUTLASS kernels at first use; vLLM pre-compiles them at build time. JIT compilation adds first-run latency but avoids build-time complexity.

2. **Fallback support**: vLLM has more extensive fallback paths (Marlin, FBGEMM, emulation, Qutlass) for pre-Blackwell GPUs. SGLang focuses primarily on Blackwell with FlashInfer fallbacks.

3. **Fused operations**: vLLM has fused SiLU+Mul+FP4 quant kernels (`silu_and_mul_nvfp4_quant`); SGLang separates these steps.

4. **MXFP4 OCP support**: Both have MXFP4 for MoE, but vLLM has broader MXFP4 linear support via CompressedTensors/Marlin/FPQuant. SGLang's MXFP4 linear falls back to BF16 dequant at load time.

5. **Backend flexibility**: SGLang exposes backend selection as a user-facing server argument; vLLM uses an internal priority-based oracle.

### The Upcasting Story

**On Blackwell (production target): No upcasting happens.** The Blackwell `tcgen05.mma.blockscaled` instruction natively consumes FP4 data with block scales. The "conversion" from FP4 to higher precision happens entirely inside the tensor core hardware as part of the multiply-accumulate operation.

**On pre-Blackwell GPUs**, different strategies are used:
- **Marlin (vLLM)**: Fuses FP4 -> FP16/BF16 dequantization into the GEMM kernel. No separate dequant pass. This is the most efficient pre-Blackwell path.
- **Hopper CUTLASS software path**: Converts E2M1 -> FP32, then uses TF32 tensor cores (19-bit effective precision).
- **SGLang MXFP4 fallback**: Dequantizes to BF16 at model load time, then runs standard BF16 GEMM. Simplest but loses all FP4 memory savings at runtime.
- **Emulation (vLLM)**: Full FP32 dequant for testing/debugging only.

### The Kernel Story

Both frameworks converge on the same underlying compute primitive: **CUTLASS 3.x `GemmUniversal` with `OpClassBlockScaledTensorOp`** targeting Blackwell tensor cores. The differences are in how they wrap, compile, and dispatch to this kernel, and what fallbacks they provide for older hardware.

For MoE workloads, both additionally support FlashInfer's TensorRT-LLM backend and CUTLASS grouped GEMM, with vLLM offering additional Triton and Marlin paths.
