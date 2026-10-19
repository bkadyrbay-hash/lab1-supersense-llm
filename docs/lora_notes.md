# QLoRA fine-tuning: setup and choices

Notebook: `notebooks/02_lora.ipynb`. Same base model as for ICL (`docs/model_choice.md`), so that the two
methods differ only in how the model is adapted to the task.

## Approach

LoRA (Hu et al., 2022) freezes the pre-trained weights and learns, for chosen linear layers, a low-rank update
`W + (alpha / r) * B A`. QLoRA (Dettmers et al., 2023) does the same on a model stored in 4-bit NF4, which is what
lets a 7B model be trained on a free 16 GB T4. We train on the whole `train` split (10,604 senses) for one epoch.

## Loading the model

The ICL notebook uses the pre-quantized `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`. Its quantization config only
keeps `lm_head` in 16-bit, so with a classification head the new `score` layer would be created directly in 4-bit
and could not be trained. We therefore load `Qwen/Qwen2.5-7B-Instruct` and quantize it at load time with the same
settings (NF4, double quantization) and `score` kept in 16-bit. The same original weights go through the
same NF4 quantization as in the unsloth checkpoint; the compute dtype is float16, since the T4 has no bfloat16 support.

## Input and target

| | generation | classification head |
|---|---|---|
| input | short English instruction + `format_input(row, with_lemma)` in the chat template | same |
| target | label name + `<|im_end|>` | index of the supersense (25 classes) |
| loss | cross-entropy on the answer tokens only (prompt tokens masked with -100) | cross-entropy on the 25 logits of the last token |
| prediction | greedy generation parsed with `parse_answer`, or label log-probability (`src/label_scoring.py`) | argmax of the 25 logits |

The prompt does not list the 25 labels with their glosses as in ICL: the fine-tuned model learns the label set
from the data, and the prompt is about 75 tokens instead of about 500, which makes an epoch much cheaper.

## Hyperparameters

Fixed for all runs (no time to tune them, they are the usual QLoRA values):

| | value |
|---|---|
| rank `r` / `lora_alpha` / dropout | 16 / 32 / 0.05 |
| learning rate, schedule | 2e-4, cosine with 3 % warm-up |
| batch size, epochs | 16, 1 |
| precision | 4-bit NF4 weights, fp16 mixed precision, LoRA weights in fp32 |
| gradient checkpointing | yes (`prepare_model_for_kbit_training`) |
| seed | 42 |

## Compared on rand-dev (one factor at a time)

1. LoRA on attention queries and values (`qv`, as in Hu et al.) vs every linear layer (`all`, as in Dettmers et al.,
   who found that adapting all layers is needed to match full fine-tuning).
2. Generation vs label log-probability with the same adapter (no retraining).
3. Generation vs classification head.
4. With vs without the lemma.

This needs four trainings. Adapters are saved on Google Drive (Colab) or as notebook output (Kaggle), never in git.

## References

- Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*.
