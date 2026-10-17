# Team workflow

This lab is done by two students. `main` always contains working code: changes reach it only
through a pull request reviewed by the other team member (direct pushes are blocked).

## Who does what

| Student A — in-context learning | Student B — LoRA fine-tuning | Both |
|---|---|---|
| `src/data.py`, `src/prompts.py`, `src/label_scoring.py` | `src/evaluate.py` | `results/metrics.csv` (append only) |
| `notebooks/00_data_exploration.ipynb`, `01_icl.ipynb` | `notebooks/02_lora.ipynb`, `03_camembert.ipynb` | `results/predictions/`, `results/figures/` |
| `docs/model_choice.md`, `docs/prompt_literature.md` | `docs/lora_notes.md`, `04_error_analysis.ipynb` (skeleton) | error analysis and method comparison |
| `README.md`, `requirements.txt` | | final report |

Each file has one owner; if you need a change in a file you do not own, ask in the pull request.

## Working on a task

1. **Branch** — on GitHub, branch menu (`main ▾`) → type `a/<task>` or `b/<task>`
   (e.g. `a/icl-baseline`, `b/lora-baseline`) → *Create branch from main*. One branch per task.
2. **Colab** — *File → Open notebook → GitHub*, paste the repository URL, pick your branch.
   The first cell clones the repository so `src/` is importable; set `BRANCH` there to use code
   that is not merged yet. New notebooks start as a copy of `notebooks/_template.ipynb`.
3. **Save the notebook** — *Edit → Clear all outputs*, then *File → Save a copy in GitHub*:
   your branch, path `notebooks/<name>.ipynb`, untick "Include a link to Colab".
4. **Other files** (`.py`, `.csv`, `.png`, `.md`) — press `.` on the repository page to open github.dev,
   or *Add file → Upload files* on your branch, or use git locally (below).
5. **Pull request** — *Compare & pull request*, fill in the template, request a review from your teammate.
6. **Review** — read the code, ask about anything unclear, approve. Then *Squash and merge*.

## Conventions

- `results/metrics.csv` is append-only and written by `log_experiment()`.
  `.gitattributes` sets `merge=union` on it, so rows added in two branches merge without conflicts.
- Prediction files are named `{method}__{config_id}__{split}.csv`, so two runs never overwrite each other.
- Dataset, model weights and LoRA adapters are not committed (`.gitignore`); adapters go to Google Drive.
- `rand-dev` is used for every tuning decision; `rand-test` and `freq-test` are run once, on the final configurations.
- All hyperparameters of a notebook live in its `CONFIG` cell; seed is 42 everywhere.
- Every pull request runs `ruff` (unused imports and variables, undefined names).
- Commit messages: short, imperative, in English — `Add zero-shot ICL baseline on rand-dev`.

## Using git locally (optional)

```bash
git switch main && git pull
git switch -c a/label-scoring
git add src/label_scoring.py
git commit -m "Add log-prob scoring of single-token labels"
git push -u origin a/label-scoring
```

If your branch falls behind `main`, click *Update branch* on the pull request page.
