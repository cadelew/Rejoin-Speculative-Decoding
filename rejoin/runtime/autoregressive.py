"""Target-only greedy reference generation."""

from typing import AbstractSet, Sequence

from rejoin.models.protocols import TargetModel
from rejoin.runtime._generation import validate_generation_request
from rejoin.types import GenerationResult


def generate_target_greedy(
    target: TargetModel,
    prefix_tokens: Sequence[int],
    max_new_tokens: int,
    stop_token_ids: AbstractSet[int] = frozenset(),
) -> GenerationResult:
    """Generate the exact target-greedy continuation token by token."""

    validate_generation_request(max_new_tokens)
    prefix = list(prefix_tokens)
    generated = []

    for _ in range(max_new_tokens):
        token = target.greedy_next(prefix)
        generated.append(token)
        prefix.append(token)
        if token in stop_token_ids:
            break

    token_count = len(generated)
    return GenerationResult(
        tokens=tuple(generated),
        steps=token_count,
        target_invocations=token_count,
        draft_tokens_proposed=0,
    )
