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
    cache, first_logprobs = out.past_key_values, out.logits[0, -1].float().log_softmax(-1)
    # The first token of each label is read at the prompt's last position. The following tokens of all labels are
    # scored in one forward pass: the prompt cache is repeated once per label, and labels are right-padded
    # (under causal attention, padding placed after a label cannot change the scores of its own tokens).
    rest = [ids[1:] for ids in labels_ids]
    longest = max(len(r) for r in rest)
    inputs = torch.tensor([ids[:-1] + [ids[-1]] * (longest - len(r)) for ids, r in zip(labels_ids, rest)],
                          device=model.device)
    cache.batch_repeat_interleave(len(labels_ids))
    logprobs = model(inputs, past_key_values=cache).logits.float().log_softmax(-1)
    return [first_logprobs[ids[0]].item() + logprobs[i, torch.arange(len(r)), torch.tensor(r)].sum().item()
            for i, (ids, r) in enumerate(zip(labels_ids, rest))]
