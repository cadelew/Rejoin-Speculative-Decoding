"""Narrow model protocols consumed by the decoding runtimes."""

from typing import Protocol, Sequence, Tuple

from rejoin.types import VerificationResult


class TargetModel(Protocol):
    """An exact target adapter.

    Production adapters should override `verify_greedy` with one batched model
    invocation. The decoder does not assume anything about KV-cache ownership.
    """

    @property
    def model_id(self) -> str:
        ...

    def greedy_next(self, prefix_tokens: Sequence[int]) -> int:
        ...

    def verify_greedy(
        self,
        prefix_tokens: Sequence[int],
        candidate_tokens: Sequence[int],
    ) -> VerificationResult:
        ...


class DraftModel(Protocol):
    """A draft adapter capable of greedy autoregressive proposals."""

    @property
    def model_id(self) -> str:
        ...

    def generate_greedy(
        self,
        prefix_tokens: Sequence[int],
        max_new_tokens: int,
    ) -> Tuple[int, ...]:
        ...
