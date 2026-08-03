"""Exact direct suffix reattachment."""

from typing import AbstractSet, Callable, Optional, Sequence

from rejoin.models.protocols import DraftModel, TargetModel
from rejoin.runtime._generation import truncate_at_stop, validate_generation_request
from rejoin.types import GenerationResult, RejoinStepResult

StepObserver = Callable[[RejoinStepResult], None]


def rejoin_exact_step(
    target: TargetModel,
    draft: DraftModel,
    prefix_tokens: Sequence[int],
    draft_length: int,
    min_suffix_length: int = 4,
) -> RejoinStepResult:
    """Run one direct-Rejoin round while preserving target-greedy exactness."""

    if draft_length <= 0:
        raise ValueError("draft_length must be positive")
    if min_suffix_length < 0:
        raise ValueError("min_suffix_length cannot be negative")

    proposed = tuple(draft.generate_greedy(prefix_tokens, draft_length))
    if not proposed:
        raise RuntimeError("draft model returned an empty proposal")
    if len(proposed) > draft_length:
        raise RuntimeError("draft model returned more tokens than requested")

    initial = target.verify_greedy(prefix_tokens, proposed)
    if initial.fully_matches:
        return RejoinStepResult(
            committed_tokens=proposed,
            draft_tokens=proposed,
            initial_verification=initial,
            rejoin_verification=None,
            attempted_rejoin=False,
            target_invocations=1,
        )

    mismatch_index = initial.greedy_match_length
    correction = initial.target_correction
    if correction is None:
        raise RuntimeError("target verification omitted the mismatch correction")

    accepted = proposed[:mismatch_index]
    suffix = proposed[mismatch_index + 1 :]
    fallback = accepted + (correction,)
    if len(suffix) < min_suffix_length:
        return RejoinStepResult(
            committed_tokens=fallback,
            draft_tokens=proposed,
            initial_verification=initial,
            rejoin_verification=None,
            attempted_rejoin=False,
            target_invocations=1,
        )

    repaired_candidate = (correction,) + suffix
    repaired_prefix = tuple(prefix_tokens) + accepted
    rejoin_verification = target.verify_greedy(repaired_prefix, repaired_candidate)
    matched = rejoin_verification.greedy_match_length

    # The correction was chosen under exactly `repaired_prefix`, so a compliant
    # target adapter must accept it. Falling back keeps third-party adapters safe.
    committed = fallback if matched == 0 else accepted + repaired_candidate[:matched]
    return RejoinStepResult(
        committed_tokens=committed,
        draft_tokens=proposed,
        initial_verification=initial,
        rejoin_verification=rejoin_verification,
        attempted_rejoin=True,
        target_invocations=2,
    )


def generate_rejoin(
    target: TargetModel,
    draft: DraftModel,
    prefix_tokens: Sequence[int],
    max_new_tokens: int,
    draft_block_size: int,
    min_suffix_length: int = 4,
    stop_token_ids: AbstractSet[int] = frozenset(),
    max_steps: int = 1_000_000,
    observer: Optional[StepObserver] = None,
) -> GenerationResult:
    """Generate an exact target-greedy continuation with direct reattachment."""

    validate_generation_request(max_new_tokens)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    prefix = list(prefix_tokens)
    generated = []
    steps = 0
    target_invocations = 0
    proposed_count = 0
    reused_count = 0

    while len(generated) < max_new_tokens:
        if steps >= max_steps:
            raise RuntimeError("Rejoin generation exceeded max_steps")
        remaining = max_new_tokens - len(generated)
        result = rejoin_exact_step(
            target=target,
            draft=draft,
            prefix_tokens=prefix,
            draft_length=min(draft_block_size, remaining),
            min_suffix_length=min_suffix_length,
        )
        if observer is not None:
            observer(result)

        committed, stopped = truncate_at_stop(
            result.committed_tokens[:remaining], stop_token_ids
        )
        generated.extend(committed)
        prefix.extend(committed)
        steps += 1
        target_invocations += result.target_invocations
        proposed_count += len(result.draft_tokens)
        # Reused tokens begin after the initially accepted prefix and correction.
        # This matters when an EOS truncates a multi-token commit before or inside
        # the retained suffix.
        reused_in_commit = max(
            0,
            len(committed) - result.initial_verification.greedy_match_length - 1,
        )
        reused_count += min(result.reused_suffix_tokens, reused_in_commit)
        if stopped:
            break

    return GenerationResult(
        tokens=tuple(generated),
        steps=steps,
        target_invocations=target_invocations,
        draft_tokens_proposed=proposed_count,
        reused_suffix_tokens=reused_count,
    )
