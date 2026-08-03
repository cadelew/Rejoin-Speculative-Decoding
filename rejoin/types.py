"""Immutable values shared by runtimes, metrics, and trace collection."""

from dataclasses import dataclass
from typing import Optional, Tuple

TokenSequence = Tuple[int, ...]


@dataclass(frozen=True)
class VerificationResult:
    """Target predictions produced while teacher-forcing a candidate sequence.

    `target_tokens[i]` is the target's greedy prediction under the prefix plus
    `candidate_tokens[:i]`. `greedy_match_length` is therefore the length of the
    longest candidate prefix that exactly follows target-greedy decoding.
    """

    candidate_tokens: TokenSequence
    target_tokens: TokenSequence
    greedy_match_length: int

    def __post_init__(self) -> None:
        if len(self.candidate_tokens) != len(self.target_tokens):
            raise ValueError("candidate_tokens and target_tokens must have equal length")
        if not 0 <= self.greedy_match_length <= len(self.candidate_tokens):
            raise ValueError("greedy_match_length is outside the candidate sequence")
        expected = _common_prefix_length(self.candidate_tokens, self.target_tokens)
        if self.greedy_match_length != expected:
            raise ValueError(
                "greedy_match_length must equal the candidate/target common prefix length"
            )

    @property
    def fully_matches(self) -> bool:
        return self.greedy_match_length == len(self.candidate_tokens)

    @property
    def target_correction(self) -> Optional[int]:
        """Return the target token at the first mismatch, if one exists."""

        if self.fully_matches:
            return None
        return self.target_tokens[self.greedy_match_length]


@dataclass(frozen=True)
class RejoinStepResult:
    """Outcome and observability data for one draft/verify round."""

    committed_tokens: TokenSequence
    draft_tokens: TokenSequence
    initial_verification: VerificationResult
    rejoin_verification: Optional[VerificationResult]
    attempted_rejoin: bool
    target_invocations: int

    def __post_init__(self) -> None:
        expected_invocations = 2 if self.attempted_rejoin else 1
        if self.target_invocations != expected_invocations:
            raise ValueError("target_invocations is inconsistent with attempted_rejoin")
        if self.attempted_rejoin != (self.rejoin_verification is not None):
            raise ValueError("attempted_rejoin and rejoin_verification are inconsistent")
        if not self.committed_tokens:
            raise ValueError("a decoding step must commit at least one token")

    @property
    def reused_suffix_tokens(self) -> int:
        if self.rejoin_verification is None:
            return 0
        return max(0, self.rejoin_verification.greedy_match_length - 1)


@dataclass(frozen=True)
class GenerationResult:
    """Tokens and aggregate counters returned by a decoding runtime."""

    tokens: TokenSequence
    steps: int
    target_invocations: int
    draft_tokens_proposed: int
    reused_suffix_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "steps",
            "target_invocations",
            "draft_tokens_proposed",
            "reused_suffix_tokens",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")


def _common_prefix_length(left: TokenSequence, right: TokenSequence) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))
