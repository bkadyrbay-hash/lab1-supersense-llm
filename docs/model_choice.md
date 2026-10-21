# Choice of the base model

We use the same base model for ICL and for LoRA fine-tuning, so that the comparison between the two methods is
not confounded by a difference in model.

## Constraints

- Instruction-tuned causal LM of at most 8B parameters (lab instructions).
- Must run on a free T4 GPU (Colab or Kaggle: 16 GB, no bfloat16), for inference with long few-shot prompts and
  for QLoRA training. For a 7–8B model this requires 4-bit quantization.
- Inputs are French definitions, so French ability matters more than English benchmark scores.

## Candidates

Scores from the [Open LLM French Leaderboard](https://huggingface.co/datasets/le-leadboard/results)
(normalized scores ×100, French translations of the Open LLM Leaderboard v2 tasks), copied from the result file of
each model, e.g. [Qwen2.5-7B-Instruct](https://huggingface.co/datasets/le-leadboard/results/blob/main/Qwen/Qwen2.5-7B-Instruct/results_2024-11-03T19-24-47.152663_norm.json),
[Llama-3.1-8B-Instruct](https://huggingface.co/datasets/le-leadboard/results/blob/main/meta-llama/Llama-3.1-8B-Instruct/results_2024-12-04T14-34-25.267808_norm.json),
[Mistral-7B-Instruct-v0.3](https://huggingface.co/datasets/le-leadboard/results/blob/main/mistralai/Mistral-7B-Instruct-v0.3/results_2025-07-27T19-28-37.498709_norm.json),
[Qwen2.5-3B-Instruct](https://huggingface.co/datasets/le-leadboard/results/blob/main/Qwen/Qwen2.5-3B-Instruct/results_2024-11-03T18-25-56.604265_norm.json)
(accessed on 18 October 2026).
MMMLU-fr (French multiple-choice knowledge questions) is the most relevant column for our task, which mostly
requires lexical and world knowledge; IFEval-fr measures how well the model follows format instructions.

| Model | Params | MMMLU-fr | BBH-fr | IFEval-fr | GPQA-fr | Average (6 tasks) |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 7.6B | 60.1 | 32.5 | 16.6 | 11.3 | 26.8 |
| Llama-3.1-8B-Instruct | 8.0B | 51.1 | 22.2 | 15.3 | 9.6 | 21.1 |
| Qwen2.5-3B-Instruct | 3.1B | 50.7 | 20.6 | 15.6 | 4.9 | 18.6 |
| Mistral-7B-Instruct-v0.3 | 7.2B | 41.3 | 19.1 | 13.8 | 6.8 | 14.6 |

The average also includes MuSR-fr and MATH Lvl5-fr (not shown).
We also tried the [European LLM Leaderboard](https://huggingface.co/spaces/Eurolingua/european-llm-leaderboard)
suggested in the lab instructions, but its result table did not load when we consulted it (17 October 2026).

## Decision

We use **Qwen2.5-7B-Instruct** in 4-bit NF4. Both methods load the pre-quantized checkpoint
[`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit)
(Apache-2.0 licence), which downloads faster and avoids quantizing at every session, so ICL and LoRA start from
exactly the same weights. For the LoRA classification head, the new `score` layer is kept in 16-bit so
that it can be trained (see `docs/lora_notes.md`).

- It has the best French scores of the candidates with at most 8B parameters, with a large margin on MMMLU-fr
  (9 points above Llama-3.1-8B).
- It is officially multilingual (29+ languages including French) and has a 32k-token context: our longest ICL
  prompt (50 demonstrations and the 25 label descriptions) is about 3.2k tokens.
- The 4-bit weights take about 5.5 GB of GPU memory, which leaves room on a T4 for long prompts at inference and
  for LoRA adapters and activations during training.

If a configuration does not fit in memory or is too slow, we fall back to Qwen2.5-3B-Instruct, which has the same
tokenizer and chat template, so the code does not change.

We did not use the newer Qwen3-8B: it is not in the French leaderboard, and its default "thinking" mode writes a
long reasoning trace before the answer, which would make generation much slower and parsing harder.

## Consequence for label scoring

In the chat template the answer starts right after `<|im_start|>assistant\n`, without a space. In that position
only 17 of the 25 supersense names are a single token for the Qwen2.5 tokenizer: `cognition`, `feeling`,
`phenomenon`, `possession`, `substance`, `groupxperson`, `act*cognition` and `artifact*cognition` are split, and
`act` / `artifact` are the first token of `act*cognition` / `artifact*cognition`.
Scoring only the first token is therefore ambiguous, so `src/label_scoring.py` scores the whole answer
(sum of the log-probabilities of the label tokens and of the end-of-turn token).
