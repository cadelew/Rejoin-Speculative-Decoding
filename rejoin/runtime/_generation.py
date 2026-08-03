"""Internal generation-loop helpers."""

from typing import AbstractSet, Sequence, Tuple


def truncate_at_stop(
    tokens: Sequence[int],
    stop_token_ids: AbstractSet[int],
) -> Tuple[Tuple[int, ...], bool]:
    """Keep tokens through the first stop token and report whether one occurred."""

    if not stop_token_ids:
        return tuple(tokens), False

    for index, token in enumerate(tokens):
        if token in stop_token_ids:
            return tuple(tokens[: index + 1]), True
    return tuple(tokens), False


def validate_generation_request(max_new_tokens: int) -> None:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
