# Cyclomatic Complexity Report: MAX Architectures

**Date:** 2026-03-30
**Tool:** [radon](https://radon.readthedocs.io/) v6.0.1 (Cyclomatic Complexity)
**Scope:** `max/python/max/pipelines/architectures/` (437 Python files)

## Summary

| Metric | Value |
|--------|-------|
| Total blocks analyzed | 2,276 (classes, functions, methods) |
| Total functions/methods | 1,785 |
| Average complexity | **A (3.28)** |
| Functions graded F (CC > 40) | 4 |
| Functions graded E (CC 31-40) | 1 |
| Functions graded D (CC 21-30) | 13 |

## Grade Distribution

| Grade | CC Range | Count | % | Description |
|-------|----------|------:|---:|-------------|
| A | 1-5 | 1,532 | 85.8% | Simple, low risk |
| B | 6-10 | 169 | 9.5% | Well-structured, moderate risk |
| C | 11-20 | 66 | 3.7% | Moderately complex |
| D | 21-30 | 13 | 0.7% | Complex, high risk |
| E | 31-40 | 1 | 0.1% | Very complex, very high risk |
| F | 41+ | 4 | 0.2% | Untestable, error-prone |

## Top 30 Most Complex Functions

| # | CC | Grade | Function | File | Line |
|--:|---:|:-----:|----------|------|-----:|
| 1 | 52 | F | `DeepseekV3_2.__call__` | `deepseekV3_2/deepseekV3_2.py` | 486 |
| 2 | 45 | F | `Qwen3VLTokenizer.new_context` | `qwen3vl_moe/tokenizer.py` | 444 |
| 3 | 42 | F | `Eagle3KimiK25Model.load_model` | `kimik2_5/unified_eagle_pipeline_model.py` | 109 |
| 4 | 41 | F | `DeepseekV3._process_hidden_states` | `deepseekV3/deepseekV3.py` | 688 |
| 5 | 34 | E | `UnifiedMTPDeepseekV3Model.load_model` | `unified_mtp_deepseekV3/model.py` | 88 |
| 6 | 29 | D | `Qwen3VLModel.execute` | `qwen3vl_moe/model.py` | 721 |
| 7 | 27 | D | `Idefics3Tokenizer.new_context` | `idefics3/tokenizer.py` | 153 |
| 8 | 25 | D | `_transform_decoder_weights` | `autoencoders/autoencoder_kl_qwen_image.py` | 717 |
| 9 | 25 | D | `Qwen3VLModel.prepare_initial_token_inputs` | `qwen3vl_moe/model.py` | 846 |
| 10 | 25 | D | `KimiK2_5Model.prepare_initial_token_inputs` | `kimik2_5/model.py` | 1058 |
| 11 | 24 | D | `Qwen2_5VLModel.prepare_initial_token_inputs` | `qwen2_5vl/model.py` | 915 |
| 12 | 23 | D | `Qwen3VLModel._build_vision_graph` | `qwen3vl_moe/model.py` | 310 |
| 13 | 22 | D | `Qwen2_5VLModel.execute` | `qwen2_5vl/model.py` | 607 |
| 14 | 21 | D | `Flux2KleinPipeline.execute` | `flux2_modulev3/pipeline_flux2_klein.py` | 170 |
| 15 | 21 | D | `Llama3.__init__` | `llama3/llama3.py` | 77 |
| 16 | 21 | D | `_convert_safetensor_with_model_config` | `llama3/weight_adapters.py` | 30 |
| 17 | 21 | D | `Qwen2_5VLModel._build_vision_graph` | `qwen2_5vl/model.py` | 281 |
| 18 | 21 | D | `DeepseekV3Model._create_model_config` | `deepseekV3/model.py` | 139 |
| 19 | 20 | C | `LlamaModelBase.execute` | `llama3/model.py` | 188 |
| 20 | 20 | C | `DistributedLlama3.__init__` | `llama3/distributed_llama.py` | 40 |
| 21 | 20 | C | `get_rope_index` | `qwen3vl_moe/nn/data_processing.py` | 24 |
| 22 | 20 | C | `DeepseekV3NextN.__call__` | `deepseekV3_nextn/deepseekV3_nextn.py` | 163 |
| 23 | 19 | C | `get_rope_index` | `qwen2_5vl/nn/data_processing.py` | 232 |
| 24 | 19 | C | `Eagle3KimiK25.__call__` | `kimik2_5/eagle3_kimi_k25.py` | 211 |
| 25 | 19 | C | `FluxPipeline.execute` | `flux1_modulev3/pipeline_flux.py` | 515 |
| 26 | 18 | C | `Olmo2.__init__` | `olmo2/olmo2.py` | 36 |
| 27 | 17 | C | `Qwen25VLEncoderModel.load_model` | `qwen2_5vl/encoder/model.py` | 81 |
| 28 | 17 | C | `T5Stack.forward` | `t5/t5.py` | 677 |
| 29 | 17 | C | `PixtralModel._load_models` | `pixtral_modulev3/model.py` | 338 |
| 30 | 17 | C | `_load_image` | `qwen3vl_moe/tokenizer.py` | 57 |

## Hotspot Architectures

Architectures with the most complex code (sum of CC for functions graded D+):

| Architecture | F-grade | D/E-grade | Highest CC | Primary concern |
|-------------|--------:|----------:|-----------:|-----------------|
| **deepseekV3_2** | 1 | 0 | 52 | Monolithic `__call__` with 52 branches |
| **deepseekV3** | 1 | 1 | 41 | `_process_hidden_states` with 41 branches |
| **qwen3vl_moe** | 1 | 3 | 45 | Tokenizer and model execution complexity |
| **kimik2_5** | 1 | 1 | 42 | Model loading logic with 42 branches |
| **unified_mtp_deepseekV3** | 0 | 1 | 34 | `load_model` with 34 branches |
| **qwen2_5vl** | 0 | 3 | 24 | Vision graph building and token preparation |
| **idefics3** | 0 | 1 | 27 | Tokenizer `new_context` |

## Methodology

**Cyclomatic Complexity (CC)** measures the number of independent paths through a function.
Each decision point (if/elif/else, for, while, and, or, except, with, assert, ternary)
adds 1 to the complexity score. A CC of 1 means a completely linear function.

| Grade | Risk Level | Recommendation |
|-------|-----------|----------------|
| A (1-5) | Low | No action needed |
| B (6-10) | Low-Moderate | Acceptable |
| C (11-20) | Moderate | Consider refactoring if modifying |
| D (21-30) | High | Should be refactored |
| E (31-40) | Very High | Strongly recommend refactoring |
| F (41+) | Critical | Untestable; refactor immediately |
