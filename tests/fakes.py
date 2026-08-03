"""Deterministic model doubles used by runtime tests."""

from typing import Callable, Sequence, Tuple

from rejoin.models.base import ReferenceTargetModel


class PositionTarget(ReferenceTargetModel):
    def __init__(self, continuation: Sequence[int], prompt_length: int = 0) -> None:
        self._continuation = tuple(continuation)
        self._prompt_length = prompt_length

    @property
    def model_id(self) -> str:
        return "position-target"

    def greedy_next(self, prefix_tokens: Sequence[int]) -> int:
        index = len(prefix_tokens) - self._prompt_length
        if not 0 <= index < len(self._continuation):
            raise IndexError("test target continuation exhausted")
        return self._continuation[index]


class FunctionDraft:
    def __init__(
        self,
        proposal_fn: Callable[[Sequence[int], int], Sequence[int]],
    ) -> None:
        self._proposal_fn = proposal_fn

    @property
    def model_id(self) -> str:
        return "function-draft"

    def generate_greedy(
        self,
        prefix_tokens: Sequence[int],
        max_new_tokens: int,
    ) -> Tuple[int, ...]:
        return tuple(self._proposal_fn(prefix_tokens, max_new_tokens))
