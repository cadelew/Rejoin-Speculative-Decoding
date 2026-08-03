"""Configuration for exact greedy Rejoin decoding."""

from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class DecoderConfig:
    """Validated controls for the reference decoder.

    `min_suffix_length` is an economic policy, not a correctness constraint.
    Reattachment is skipped below this length because a second target
    verification is unlikely to repay its overhead.
    """

    draft_block_size: int = 16
    min_suffix_length: int = 4
    stop_token_ids: FrozenSet[int] = frozenset()
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.draft_block_size <= 0:
            raise ValueError("draft_block_size must be positive")
        if self.min_suffix_length < 0:
            raise ValueError("min_suffix_length cannot be negative")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
