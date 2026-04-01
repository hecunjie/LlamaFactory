from typing import Tuple

from transformers import PreTrainedModel, PreTrainedTokenizerBase


PAUSE_TOKEN = "[PAUSE]"


def register_pause_token(
    tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel
) -> Tuple[int, int]:
    """Register [PAUSE] token and resize embeddings when needed."""
    if tokenizer.convert_tokens_to_ids(PAUSE_TOKEN) != tokenizer.unk_token_id:
        pause_token_id = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)
        return pause_token_id, 0

    added = tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    pause_token_id = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)
    return pause_token_id, added
