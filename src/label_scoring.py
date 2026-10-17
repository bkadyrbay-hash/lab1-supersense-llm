"""Classification by label log-probabilities: the model scores each possible answer instead of generating one."""

import torch

from src.data import SUPERSENSES
from src.prompts import LABEL_NAMES


def label_token_ids(tokenizer, lang: str) -> list[list[int]]:
    """Token ids of each answer as the model writes it: the label name, then the end-of-turn token.

    The end-of-turn token makes the scores comparable for labels that are prefixes of other labels
    ("act" vs "act*cognition"): only the full label followed by the end of the answer gets the probability.
    """
    # For Qwen2.5-Instruct the eos token is <|im_end|>, which closes every assistant turn.
    return [tokenizer(LABEL_NAMES[lang][s], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            for s in SUPERSENSES]


@torch.no_grad()
def score_labels(model, input_ids: torch.Tensor, labels_ids: list[list[int]], cache=None) -> list[float]:
    """Log-probability of each label's tokens right after the prompt `input_ids` (shape 1 x length).

    `cache` may already hold the keys/values of the beginning of the prompt; it is extended in place.
    """
    cached = cache.get_seq_length() if cache is not None else 0
    out = model(input_ids[:, cached:], past_key_values=cache, use_cache=True)
    cache, first_logprobs = out.past_key_values, out.logits[0, -1].log_softmax(-1)
    prompt_len = input_ids.shape[1]
    scores = []
    for ids in labels_ids:
        # First token: read from the prompt's last position; following tokens: one short forward pass.
        rest = torch.tensor(ids[1:], device=model.device)
        logits = model(torch.tensor([ids[:-1]], device=model.device), past_key_values=cache, use_cache=True).logits[0]
        scores.append(first_logprobs[ids[0]].item() + logits.log_softmax(-1).gather(1, rest[:, None]).sum().item())
        cache.crop(prompt_len)  # forget this label's tokens before scoring the next one
    return scores
