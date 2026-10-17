# Choice of the base model

We use the **same base model for ICL and for LoRA fine-tuning**, so that the comparison between the two
methods is not confounded by a difference in model.

## Constraints

- Instruction-tuned causal LM of **at most 8B parameters** (lab instructions).
- Must run on a free **Colab T4 GPU (16 GB, no bf16)**, for inference with long few-shot prompts *and* for
  QLoRA training: this requires 4-bit quantization for a 7–8B model.
- Inputs are **French** definitions, so French ability matters more than English benchmark scores.

## Candidates

Scores from the [Open LLM French Leaderboard](https://huggingface.co/datasets/le-leadboard/results)
(normalized scores, ×100; French translations of the Open LLM Leaderboard v2 tasks).
MMMLU-fr (French multiple-choice knowledge questions) is the most relevant column for our task, which mostly
requires lexical and world knowledge; IFEval-fr measures how well the model follows format instructions.

| Model | Params | MMMLU-fr | BBH-fr | IFEval-fr | GPQA-fr | Average (6 tasks) |
|---|---|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** | 7.6B | **60.1** | **32.5** | **16.6** | **11.3** | **26.8** |
| Llama-3.1-8B-Instruct | 8.0B | 51.1 | 22.2 | 15.3 | 9.6 | 21.1 |
| Qwen2.5-3B-Instruct | 3.1B | 50.7 | 20.6 | 15.6 | 4.9 | 18.6 |
| Mistral-7B-Instruct-v0.3 | 7.2B | 41.3 | 19.1 | 13.8 | 6.8 | 14.6 |

(The average also includes MuSR-fr and MATH Lvl5-fr, not shown.)
We also tried the [European LLM Leaderboard](https://huggingface.co/spaces/Eurolingua/european-llm-leaderboard)
suggested in the lab instructions, but its result table did not load when we consulted it (October 2026).

## Decision

**`Qwen2.5-7B-Instruct`**, loaded in 4-bit NF4 through the pre-quantized
[`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit) checkpoint
(Apache-2.0 licence).

- Best French scores of the candidates of ≤ 8B parameters, by a large margin on MMMLU-fr (+9 points over Llama-3.1-8B).
- Officially multilingual (29+ languages including French), 32k-token context: a prompt with 50 demonstrations
  plus 25 label descriptions (≈ 3–4k tokens) fits easily.
- The 4-bit weights take about 5.5 GB of GPU memory, which leaves room on a T4 for long prompts at inference
  and for LoRA adapters and activations during training.
- Pre-quantized weights download faster and avoid quantizing at every Colab session; the model is widely used
  with `peft`/`bitsandbytes`, so QLoRA examples are easy to adapt.

**Fallback:** `Qwen2.5-3B-Instruct` (same tokenizer and chat template, so the code does not change) if a
configuration does not fit in memory or is too slow.

**Not chosen:** the newer `Qwen3-8B` is not yet in the French leaderboard, and its default "thinking" mode
produces long reasoning traces before the answer, which would make generation much slower and parsing harder;
we preferred a well-documented model with published French scores.

## Consequence for label scoring

With the Qwen2.5 tokenizer, 22 of the 25 supersense names are a single token when preceded by a space
(`" person"`, `" artifact"`…), but `act*cognition`, `artifact*cognition` and `groupxperson` are split into several
tokens, and `act` / `artifact` are the first token of `act*cognition` / `artifact*cognition`.
Scoring only the first token is therefore ambiguous; label log-probabilities must be computed over the full
label (sum of token log-probs) or with single-token codes (see `src/label_scoring.py`).
