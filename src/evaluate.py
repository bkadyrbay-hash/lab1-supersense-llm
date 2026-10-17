"""Metrics, prediction files and the experiment log shared by all methods."""

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METRICS_COLUMNS = [
    "date", "author", "method", "config_id", "split", "model", "with_lemma", "prompt_lang", "n_shots",
    "shot_selection", "lora_targets", "head", "decoding", "epochs", "accuracy_super", "macro_f1_super",
    "accuracy_hyper", "macro_f1_hyper", "invalid_rate", "n", "runtime_min", "notes",
]


def evaluate(gold: list[str], pred: list[str | None], labels: list[str]) -> dict:
    """Accuracy, macro-F1, per-class F1 and share of predictions outside `labels` (None = unparsable answer)."""
    pred = [p if p in labels else "INVALID" for p in pred]
    # Macro-F1 over the classes present in the gold data: rand-dev has no 'relation' instance, and a class
    # absent from gold would add an F1 of 0 that says nothing about the classifier.
    present = sorted(set(gold))
    per_class = f1_score(gold, pred, labels=present, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": float(per_class.mean()),
        "n": len(gold),
        "invalid_rate": pred.count("INVALID") / len(pred),
        "per_class": dict(zip(present, per_class.round(4).tolist())),
    }


def paired_bootstrap(gold: list[str], pred_a: list, pred_b: list, labels: list[str], n_samples: int = 1000,
                     seed: int = 42) -> float:
    """Share of bootstrap resamples of the evaluation set where system B has a higher macro-F1 than A."""
    rng = np.random.default_rng(seed)
    gold, pred_a, pred_b = map(np.asarray, (gold, pred_a, pred_b))
    wins = 0
    for _ in range(n_samples):
        idx = rng.integers(0, len(gold), len(gold))
        f1_a = evaluate(gold[idx].tolist(), pred_a[idx].tolist(), labels)["macro_f1"]
        f1_b = evaluate(gold[idx].tolist(), pred_b[idx].tolist(), labels)["macro_f1"]
        wins += f1_b > f1_a
    return wins / n_samples


def save_predictions(df_pred: pd.DataFrame, method: str, config_id: str, split: str) -> str:
    """Write one run's predictions to results/predictions/{method}__{config_id}__{split}.csv."""
    path = RESULTS_DIR / "predictions" / f"{method}__{config_id}__{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(path, index=False)
    return str(path)


def log_experiment(row: dict) -> None:
    """Append one run to results/metrics.csv (unknown fields stay empty)."""
    path = RESULTS_DIR / "metrics.csv"
    row = {"date": date.today().isoformat(), **row}
    pd.DataFrame([row], columns=METRICS_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def plot_confusion(gold: list[str], pred: list, labels: list[str], path: str) -> None:
    """Row-normalised confusion matrix (each row = one gold class), saved to `path`."""
    pred = [p if p in labels else "INVALID" for p in pred]
    cols = labels + (["INVALID"] if "INVALID" in pred else [])
    cm = confusion_matrix(gold, pred, labels=cols)[: len(labels)]
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.35 * len(cols) + 2, 0.35 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
