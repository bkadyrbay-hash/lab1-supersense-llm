"""Loading the labeled word senses and formatting them as model inputs."""

from pathlib import Path
import urllib.request

import pandas as pd

DATA_URL = (
    "http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/"
    "super_wiktionary_labeled_data_simplified.noex.25labels.tsv.gz"
)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / DATA_URL.rsplit("/", 1)[-1]

# Supersense -> hypersense mapping (taxonomy of Barque et al., 2020), copied from the lab notebook.
SUPER2HYPER = {
    "artifact": "inanimate",
    "body": "inanimate",
    "food": "inanimate",
    "object": "inanimate",
    "plant": "inanimate",
    "substance": "inanimate",
    "act": "dynamic_situation",
    "event": "dynamic_situation",
    "phenomenon": "dynamic_situation",
    "animal": "animate",
    "person": "animate",
    "groupxperson": "animate",
    "cognition": "information",
    "communication": "information",
    "quantity": "quantity",
    "attribute": "stative_situation",
    "feeling": "stative_situation",
    "relation": "stative_situation",
    "state": "stative_situation",
    "time": "time",
    "possession": "possession",
    "institution": "institution",
    "act*cognition": "dynamic_situation*information",
    "artifact*cognition": "inanimate*information",
    "other": "other",
}
SUPERSENSES = sorted(SUPER2HYPER)
HYPERSENSES = sorted(set(SUPER2HYPER.values()))


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset (downloading it on first use), add 'hypersense', drop protect-frsemcor splits."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path, sep="\t")
    # The lab instructions say to ignore both protect-frsemcor-dev and protect-frsemcor-test.
    df = df[~df["set"].str.startswith("protect-frsemcor")].reset_index(drop=True)
    # 19 senses have no definition; we keep them (dropping test items would bias evaluation).
    df["definition"] = df["definition"].fillna("")
    df["hypersense"] = df["supersense"].map(SUPER2HYPER)
    return df


def get_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return the rows of one split: 'train', 'rand-dev', 'freq-dev', 'rand-test' or 'freq-test'."""
    if name not in set(df["set"]):
        raise ValueError(f"Unknown split {name!r}, expected one of {sorted(set(df['set']))}")
    return df[df["set"] == name].reset_index(drop=True)


def format_input(row: pd.Series, with_lemma: bool) -> str:
    """Render one sense as the text the model sees (French field names, like the data)."""
    if with_lemma:
        return f"Lemme : {row['lemma']}\nDéfinition : {row['definition']}"
    return f"Définition : {row['definition']}"
