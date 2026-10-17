"""Chat prompts for classifying word senses into supersenses with in-context learning (ICL)."""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.data import SUPERSENSES, format_input

# Label names shown to the model. In "fr" prompts everything, labels included, is in French.
LABEL_NAMES = {
    "en": {s: s for s in SUPERSENSES},
    "fr": {
        "act": "acte",
        "act*cognition": "acte*cognition",
        "animal": "animal",
        "artifact": "artefact",
        "artifact*cognition": "artefact*cognition",
        "attribute": "attribut",
        "body": "corps",
        "cognition": "cognition",
        "communication": "communication",
        "event": "événement",
        "feeling": "sentiment",
        "food": "nourriture",
        "groupxperson": "groupe_humain",
        "institution": "institution",
        "object": "objet",
        "other": "autre",
        "person": "personne",
        "phenomenon": "phénomène",
        "plant": "plante",
        "possession": "possession",
        "quantity": "quantité",
        "relation": "relation",
        "state": "état",
        "substance": "substance",
        "time": "temps",
    },
}

# One-line glosses: some names are opaque without them ("groupxperson", "other", the "*" complex types).
GLOSSES = {
    "en": {
        "act": "an action or activity performed by an agent (negotiation, jump, cleaning)",
        "act*cognition": "an act that is also its mental or informational content (judgement, remark, explanation)",
        "animal": "an animal or a kind of animal",
        "artifact": "a man-made physical object (tool, building, vehicle, garment)",
        "artifact*cognition": "a physical object that carries information (book, letter, map, script)",
        "attribute": "a property or quality of an entity (strength, flexibility, size)",
        "body": "a part of a human or animal body, or a bodily substance",
        "cognition": "a mental content or process (knowledge, idea, belief, field of study)",
        "communication": "a message or communicative content (speech, word, signal)",
        "event": "something that happens without being an agent's action (accident, birth)",
        "feeling": "an emotion or feeling (joy, fear)",
        "food": "food or drink",
        "groupxperson": "a group of people (team, population, staff)",
        "institution": "an organisation or institution (company, school, government)",
        "object": "a natural non-living physical object or place (stone, river, star)",
        "other": "a sense that fits none of the other classes",
        "person": "a human being, incl. roles, professions and inhabitants of a place",
        "phenomenon": "a natural phenomenon or process (rain, magnetism)",
        "plant": "a plant or a kind of plant",
        "possession": "money or an asset that can be owned or transferred (price, salary, inheritance)",
        "quantity": "a quantity, a measure or a unit of measurement",
        "relation": "a relation between entities (link, difference, order)",
        "state": "a state or condition (illness, situation, anonymity)",
        "substance": "a material or substance (water, iron, chemical compound)",
        "time": "a time period or point in time (day, era, moment)",
    },
    "fr": {
        "act": "une action ou activité réalisée par un agent (négociation, saut, nettoyage)",
        "act*cognition": "un acte qui est aussi son contenu mental ou informationnel (jugement, remarque, explication)",
        "animal": "un animal ou une espèce animale",
        "artifact": "un objet physique fabriqué par l'homme (outil, bâtiment, véhicule, vêtement)",
        "artifact*cognition": "un objet physique porteur d'information (livre, lettre, carte, scénario)",
        "attribute": "une propriété ou qualité d'une entité (force, souplesse, taille)",
        "body": "une partie du corps humain ou animal, ou une substance corporelle",
        "cognition": "un contenu ou processus mental (savoir, idée, croyance, discipline)",
        "communication": "un message ou contenu communicatif (discours, mot, signal)",
        "event": "quelque chose qui arrive sans être l'action d'un agent (accident, naissance)",
        "feeling": "une émotion ou un sentiment (joie, peur)",
        "food": "un aliment ou une boisson",
        "groupxperson": "un groupe de personnes (équipe, population, personnel)",
        "institution": "une organisation ou institution (entreprise, école, gouvernement)",
        "object": "un objet physique naturel non vivant ou un lieu (pierre, rivière, étoile)",
        "other": "un sens qui ne correspond à aucune autre classe",
        "person": "un être humain, y compris rôles, professions et habitants d'un lieu",
        "phenomenon": "un phénomène ou processus naturel (pluie, magnétisme)",
        "plant": "une plante ou une espèce végétale",
        "possession": "de l'argent ou un bien que l'on possède ou transfère (prix, salaire, héritage)",
        "quantity": "une quantité, une mesure ou une unité de mesure",
        "relation": "une relation entre entités (lien, différence, ordre)",
        "state": "un état ou une condition (maladie, situation, anonymat)",
        "substance": "une matière ou substance (eau, fer, composé chimique)",
        "time": "une période ou un moment (jour, époque, instant)",
    },
}

INSTRUCTIONS = {
    "en": (
        "You are an expert lexicographer. You will be given a sense of a French noun, described by {source} "
        "from the French Wiktionary. Classify this sense into exactly one of the following semantic classes "
        "(supersenses):\n{labels}\n\nAnswer with the class name only, exactly as written above."
    ),
    "fr": (
        "Vous êtes un lexicographe expert. On vous donne un sens d'un nom français tiré du Wiktionnaire, "
        "décrit par {source}. Classez ce sens dans exactement une des classes sémantiques (supersens) "
        "suivantes :\n{labels}\n\nRépondez uniquement par le nom de la classe, exactement comme écrit ci-dessus."
    ),
}
SOURCES = {
    "en": {True: "its lemma and its definition", False: "its definition"},
    "fr": {True: "son lemme et sa définition", False: "sa définition"},
}


class ShotSelector:
    """Chooses the training examples shown as demonstrations before a query."""

    def __init__(self, train_df: pd.DataFrame, n_shots: int, strategy: str, seed: int = 42):
        """strategy: 'random' or 'by_class' (same shots for every query) or 'similar' (per query)."""
        pool = train_df[train_df["definition"] != ""].reset_index(drop=True)  # empty definitions teach nothing
        self.n_shots, self.strategy = n_shots, strategy
        if strategy == "random":
            self.fixed = pool.sample(n_shots, random_state=seed)
        elif strategy == "by_class":
            # Balanced labels in a shuffled order, against majority and recency bias (Zhao et al., 2021):
            # one shot per class and per round, classes in random order (5 shots = 5 different classes).
            classes = np.random.default_rng(seed).permutation(SUPERSENSES)
            n_rounds = -(-n_shots // len(classes))
            per_class = {c: pool[pool["supersense"] == c].sample(n_rounds, random_state=seed) for c in classes}
            rows = [per_class[c].iloc[r] for r in range(n_rounds) for c in classes]
            self.fixed = pd.DataFrame(rows[:n_shots])
        elif strategy == "similar":
            # Nearest training definitions (Liu et al., 2022); character n-grams cope with French morphology.
            self.pool = pool
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(pool["definition"])
        else:
            raise ValueError(f"Unknown shot selection strategy {strategy!r}")

    def __call__(self, row: pd.Series) -> pd.DataFrame:
        """Return the demonstrations for the query `row`."""
        if self.strategy != "similar":
            return self.fixed
        scores = linear_kernel(self.vectorizer.transform([row["definition"]]), self.matrix).ravel()
        # Most similar shot last, i.e. right before the query.
        return self.pool.iloc[np.argsort(-scores)[: self.n_shots][::-1]]


def system_prompt(lang: str, with_lemma: bool) -> str:
    """Task instruction listing every label with its gloss."""
    labels = "\n".join(f"- {LABEL_NAMES[lang][s]}: {GLOSSES[lang][s]}" for s in SUPERSENSES)
    return INSTRUCTIONS[lang].format(source=SOURCES[lang][with_lemma], labels=labels)


def build_prompt(row: pd.Series, shots: pd.DataFrame, lang: str, with_lemma: bool) -> list[dict]:
    """Chat messages: instruction, then each shot as a user/assistant turn, then the query."""
    messages = [{"role": "system", "content": system_prompt(lang, with_lemma)}]
    for _, shot in shots.iterrows():
        messages.append({"role": "user", "content": format_input(shot, with_lemma)})
        messages.append({"role": "assistant", "content": LABEL_NAMES[lang][shot["supersense"]]})
    messages.append({"role": "user", "content": format_input(row, with_lemma)})
    return messages


def parse_answer(text: str, lang: str) -> str | None:
    """Map a generated answer back to an English supersense, or None if no label is recognised."""
    name2label = {name: label for label, name in LABEL_NAMES[lang].items()}
    answer = text.strip().lower().strip(" .:*\"'`")  # '*' for markdown bold
    if answer in name2label:
        return name2label[answer]
    # Otherwise take the longest label name occurring as a whole word, so "act*cognition" beats "act".
    found = [name for name in name2label if re.search(rf"(?<![\w*]){re.escape(name)}(?![\w*])", answer)]
    return name2label[max(found, key=len)] if found else None
