"""Ordinary exact greedy speculative decoding baseline."""

from typing import AbstractSet, Sequence, Tuple

from rejoin.models.protocols import DraftModel, TargetModel
from rejoin.runtime._generation import truncate_at_stop, validate_generation_request
from rejoin.types import GenerationResult, VerificationResult


def speculative_step(
    target: TargetModel,
    draft: DraftModel,
    prefix_tokens: Sequence[int],
    draft_length: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], VerificationResult]:
    """Run one ordinary speculative round and return committed tokens.

    This conservative baseline does not request a target bonus token after a
    fully accepted draft block. That affects performance, not output exactness.
    """

    if draft_length <= 0:
        raise ValueError("draft_length must be positive")

    proposed = tuple(draft.generate_greedy(prefix_tokens, draft_length))
    if not proposed:
        raise RuntimeError("draft model returned an empty proposal")
    if len(proposed) > draft_length:
        raise RuntimeError("draft model returned more tokens than requested")

    verification = target.verify_greedy(prefix_tokens, proposed)
    matched = verification.greedy_match_length

    if verification.fully_matches:
        return proposed, proposed, verification

    correction = verification.target_correction
    if correction is None:  # Defensive guard for malformed third-party adapters.
        raise RuntimeError("target verification omitted the mismatch correction")
    committed = proposed[:matched] + (correction,)
    return committed, proposed, verification


def generate_speculative(
    target: TargetModel,
    draft: DraftModel,
    prefix_tokens: Sequence[int],
    max_new_tokens: int,
    draft_block_size: int,
    stop_token_ids: AbstractSet[int] = frozenset(),
    max_steps: int = 1_000_000,
) -> GenerationResult:
    """Generate an exact target-greedy continuation using ordinary speculation."""

    validate_generation_request(max_new_tokens)
    if draft_block_size <= 0:
        raise ValueError("draft_block_size must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    prefix = list(prefix_tokens)
    generated = []
    steps = 0
    proposed_count = 0

    while len(generated) < max_new_tokens:
        if steps >= max_steps:
            raise RuntimeError("speculative generation exceeded max_steps")
        remaining = max_new_tokens - len(generated)
        committed, proposed, _ = speculative_step(
            target=target,
            draft=draft,
            prefix_tokens=prefix,
            draft_length=min(draft_block_size, remaining),
        )
        committed, stopped = truncate_at_stop(committed[:remaining], stop_token_ids)
        generated.extend(committed)
        prefix.extend(committed)
        proposed_count += len(proposed)
        steps += 1
        if stopped:
            break

    return GenerationResult(
        tokens=tuple(generated),
        steps=steps,
        target_invocations=steps,
        draft_tokens_proposed=proposed_count,
    )
