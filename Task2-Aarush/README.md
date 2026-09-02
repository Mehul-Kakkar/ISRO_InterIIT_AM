# Task 2 — Satellite Image Captioning

## Overview

This task implements a satellite-image captioning pipeline using an open-source
vision-language model and evaluates the generated captions against the provided
VRSBench reference captions using BLEU scores.

The evaluation is performed on the six images specified in the task.

## Model

**Model:** Qwen2.5-VL-7B-Instruct  
**Source:** Qwen  
**License:** Apache-2.0

The model is run locally using Hugging Face Transformers with 4-bit NF4
quantization to reduce GPU memory usage.

No hosted commercial vision-language APIs such as GPT-4V, Gemini, or Claude
are used.

## Dataset

The task uses the **VRSBench** dataset and its provided captioning evaluation
annotations.

Reference captions are obtained from:

`VRSBench_EVAL_Cap.json`

The six evaluated images are:

- `P0003_0002.png`
- `P0019_0002.png`
- `P0060_0004.png`
- `P0110_0017.png`
- `P0146_0005.png`
- `P0168_0009.png`

## Pipeline

The pipeline consists of the following steps:

1. Load Qwen2.5-VL-7B-Instruct locally in 4-bit quantized mode.
2. Load each of the six satellite images.
3. Generate a descriptive caption using a fixed prompting strategy.
4. Save the generated captions to `outputs/generated_captions.txt`.
5. Compare the generated captions with the VRSBench ground-truth captions.
6. Compute BLEU-1, BLEU-2, BLEU-3, and BLEU-4 for each image.
7. Compute aggregate corpus BLEU scores.

## Requirements

Python 3.10+ is recommended.

Install the required packages:

```bash
pip install torch
pip install transformers
pip install accelerate
pip install bitsandbytes
pip install qwen-vl-utils
pip install pillow
pip install nltk
