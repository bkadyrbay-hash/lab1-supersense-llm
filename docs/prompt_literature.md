# Prompting LLMs for text classification: what we take from the literature

## The task

Assigning a coarse semantic class to a noun sense is the *supersense* problem introduced for WordNet nouns
by Ciaramita & Johnson (2003), who already framed it as classifying senses of unknown nouns. Our setting
differs in that each sense is given by its dictionary definition (Angleraud et al., 2025), with the taxonomy of
Barque et al. (2020). An important property of the data (see `00_data_exploration`): no lemma of a dev/test set
appears in the training set (0 % overlap), so demonstrations can never contain the same word as the query.

## Findings we use

| # | Source | Finding | What we do |
|---|---|---|---|
| 1 | Brown et al. (2020) | Instruction + a few input/label demonstrations lets a model perform a task without training; accuracy grows with the number of shots. | Compare 0 / 5 / 25 / 50 demonstrations. |
| 2 | Min et al. (2022) | Demonstrations mostly help by showing the label space, the input distribution and the output format; whether their labels are correct matters surprisingly little. | Always list the 25 labels with a short gloss in the instruction, so even 0-shot sees the full label space; give demonstrations as chat turns (user = input, assistant = label only) to fix the format. |
| 3 | Zhao et al. (2021) | Few-shot predictions are biased towards labels that are frequent or recent in the prompt (majority / recency bias). | `by_class` selection: demonstrations balanced over labels, in a shuffled order (seeded), instead of a random sample dominated by frequent classes. |
| 4 | Lu et al. (2022) | Performance varies a lot with the order of the same demonstrations. | Keep a fixed seed so all configurations of a comparison use the same order; report it as a limitation (no time to average over orders). |
| 5 | Gao et al. (2021); Liu et al. (2022) | Choosing demonstrations similar to the query (nearest neighbours in an embedding space) beats random demonstrations. | Our addition: a `similar` selection that picks, for each query, the training definitions closest to its definition (TF-IDF over character n-grams: cheap, no GPU). |
| 6 | Lai et al. (2023) | For non-English inputs, ChatGPT performs better or comparably with English task descriptions than with descriptions in the target language. | Compare `en` (English instruction + English labels) vs `fr` (French instruction + French label names) for the same French inputs. |
| 7 | Holtzman et al. (2021) | Scoring answers by their probability suffers from *surface form competition* (several strings for the same answer, unequal lengths). | For label scoring, use the full-label log-probability and keep labels short and unambiguous; compare with free generation. |
| 8 | Sun et al. (2023) | Asking the model for clues and reasoning before the label (CARP) improves LLM text classification. | Not used by default (much slower generation on a T4); possible extension if time allows. |

## Prompt design (implemented in `src/prompts.py`)

```
system:    You are an expert lexicographer. You will be given a sense of a French noun, described by its lemma
           and its definition from the French Wiktionary. Classify this sense into exactly one of the following
           semantic classes (supersenses):
           - act: an action or activity performed by an agent (negotiation, jump, cleaning)
           - ...   (25 labels, each with a one-line gloss)
           Answer with the class name only, exactly as written above.
user:      Lemme : <lemma>
           Définition : <definition>        (one user/assistant pair per demonstration)
assistant: <label>
user:      Lemme : <lemma>
           Définition : <definition>        (the query)
```

Factors compared on `rand-dev`, one at a time, each keeping the best value of the previous ones:

1. prompt language (`en` / `fr`);
2. number of shots (0 / 5 / 25 / 50);
3. shot selection (`random` / `by_class` / `similar`);
4. with or without the lemma;
5. generation vs label log-probability.

## References

- Angleraud, N., Barque, L., Candito, M. (2025). Annotating the French Wiktionary with supersenses for large scale lexical analysis: a use case to assess form-meaning relationships within the nominal lexicon. *COLING 2025*.
- Barque, L., Haas, P., Huyghe, R., et al. (2020). FrSemCor: Annotating a French corpus with supersenses. *LREC 2020*.
- Brown, T. et al. (2020). Language Models are Few-Shot Learners. *NeurIPS 2020*.
- Ciaramita, M., Johnson, M. (2003). Supersense Tagging of Unknown Nouns in WordNet. *EMNLP 2003*.
- Gao, T., Fisch, A., Chen, D. (2021). Making Pre-trained Language Models Better Few-shot Learners. *ACL 2021*.
- Holtzman, A., West, P., Shwartz, V., Choi, Y., Zettlemoyer, L. (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. *EMNLP 2021*.
- Lai, V. D. et al. (2023). ChatGPT Beyond English: Towards a Comprehensive Evaluation of Large Language Models in Multilingual Learning. *Findings of EMNLP 2023*.
- Liu, J. et al. (2022). What Makes Good In-Context Examples for GPT-3? *DeeLIO Workshop, ACL 2022*.
- Lu, Y. et al. (2022). Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity. *ACL 2022*.
- Min, S. et al. (2022). Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? *EMNLP 2022*.
- Sun, X. et al. (2023). Text Classification via Large Language Models. *Findings of EMNLP 2023*.
- Zhao, Z., Wallace, E., Feng, S., Klein, D., Singh, S. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. *ICML 2021*.
