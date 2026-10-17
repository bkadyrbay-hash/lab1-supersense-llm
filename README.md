# Classifying word-sense definitions into supersenses with LLMs

M2 Computational Linguistics — *NLP for Society* (Marie Candito) — Lab 1.

We classify French Wiktionary word senses (lemma + definition) into 25 **supersenses**
(and, through a fixed mapping, 12 **hypersenses**) and compare:

1. **In-context learning (ICL)** with a 4-bit instruction-tuned causal LLM (`Qwen2.5-7B-Instruct`);
2. **QLoRA fine-tuning** of the same model on the training set (generation vs classification head);
3. *(optional)* full fine-tuning of an encoder (CamemBERT).

Data: [Angleraud et al., 2025](https://aclanthology.org/2025.coling-main.356/), taxonomy from
[Barque et al., 2020](https://aclanthology.org/2020.lrec-1.724/).
Hyperparameters are tuned on `rand-dev` only; final results are reported once on `rand-test` and `freq-test`.

## Notebooks (open in Colab)

| Notebook | Content | Colab |
|---|---|---|
| `00_data_exploration` | label distributions per split, input properties | [open](https://colab.research.google.com/github/bkadyrbay-hash/lab1-supersense-llm/blob/main/notebooks/00_data_exploration.ipynb) |
| `01_icl` | prompting experiments (language, #shots, shot selection, lemma, generation vs label log-prob) | [open](https://colab.research.google.com/github/bkadyrbay-hash/lab1-supersense-llm/blob/main/notebooks/01_icl.ipynb) |
| `02_lora` | QLoRA: target modules, generation vs classification head, lemma | [open](https://colab.research.google.com/github/bkadyrbay-hash/lab1-supersense-llm/blob/main/notebooks/02_lora.ipynb) |
| `03_camembert` | optional encoder baseline | [open](https://colab.research.google.com/github/bkadyrbay-hash/lab1-supersense-llm/blob/main/notebooks/03_camembert.ipynb) |
| `04_error_analysis` | confusion matrices, per-class F1, errors shared by methods | [open](https://colab.research.google.com/github/bkadyrbay-hash/lab1-supersense-llm/blob/main/notebooks/04_error_analysis.ipynb) |

Every notebook starts with a setup cell that clones this repository in Colab, so `src/` is importable
and the dataset is downloaded automatically on first use. Use a GPU runtime (T4) for `01`–`03`.

## Repository layout

```
src/data.py            loading, splits (protect-frsemcor removed), input formatting
src/evaluate.py        metrics (accuracy, macro-F1), prediction files, experiment log
src/prompts.py         ICL prompt templates
src/label_scoring.py   label log-probabilities after the prompt
notebooks/             experiments (one notebook per method)
results/metrics.csv    one row per run — the single table all report numbers come from
results/predictions/   one CSV of predictions per run, used for error analysis
results/figures/       figures of the report
docs/                  short notes justifying our choices (model, prompts, LoRA)
```

## Running locally

```bash
pip install -r requirements.txt
jupyter lab notebooks/00_data_exploration.ipynb
```

Team workflow (branches, pull requests, committing from Colab): see [CONTRIBUTING.md](CONTRIBUTING.md).
