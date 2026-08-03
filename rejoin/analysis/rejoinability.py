"""Direct suffix-survival measurement for Phase 1 experiments."""

from dataclasses import dataclass
from typing import Optional, Sequence

from rejoin.models.protocols import DraftModel, TargetModel


@dataclass(frozen=True)
class RejoinabilityResult:
    fully_accepted: bool
    draft_length: int
    mismatch_index: Optional[int]
    correction_token: Optional[int]
    suffix_length: int
    direct_survival_length: int

    @property
    def suffix_salvage_ratio(self) -> float:
        if self.suffix_length == 0:
            return 0.0
        return self.direct_survival_length / self.suffix_length


def measure_direct_rejoinability(
    target: TargetModel,
    draft: DraftModel,
    prefix_tokens: Sequence[int],
    draft_length: int,
) -> RejoinabilityResult:
    """Teacher-force a retained suffix after inserting the target correction."""

    if draft_length <= 0:
        raise ValueError("draft_length must be positive")
    proposed = tuple(draft.generate_greedy(prefix_tokens, draft_length))
    if not proposed:
        raise RuntimeError("draft model returned an empty proposal")
    if len(proposed) > draft_length:
        raise RuntimeError("draft model returned more tokens than requested")

    initial = target.verify_greedy(prefix_tokens, proposed)
    if initial.fully_matches:
        return RejoinabilityResult(
            fully_accepted=True,
            draft_length=len(proposed),
            mismatch_index=None,
            correction_token=None,
            suffix_length=0,
            direct_survival_length=0,
        )

    mismatch_index = initial.greedy_match_length
    correction = initial.target_correction
    if correction is None:
        raise RuntimeError("target verification omitted the mismatch correction")
    suffix = proposed[mismatch_index + 1 :]

    if not suffix:
        survival = 0
    else:
        repaired_prefix = tuple(prefix_tokens) + proposed[:mismatch_index] + (correction,)
        survival = target.verify_greedy(repaired_prefix, suffix).greedy_match_length

    return RejoinabilityResult(
        fully_accepted=False,
        draft_length=len(proposed),
        mismatch_index=mismatch_index,
        correction_token=correction,
        suffix_length=len(suffix),
        direct_survival_length=survival,
    )
