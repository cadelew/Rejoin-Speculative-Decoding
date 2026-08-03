"""Correctness-oriented base classes for model integrations."""

from abc import ABC, abstractmethod
from typing import Sequence

from rejoin.types import VerificationResult


class ReferenceTargetModel(ABC):
    """Build target verification from a next-token primitive.

    This implementation invokes `greedy_next` once per candidate token. It is
    useful for tests and CPU reference models. GPU integrations should implement
    `verify_greedy` as a batched teacher-forcing call instead.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def greedy_next(self, prefix_tokens: Sequence[int]) -> int:
        raise NotImplementedError

    def verify_greedy(
        self,
        prefix_tokens: Sequence[int],
        candidate_tokens: Sequence[int],
    ) -> VerificationResult:
        prefix = list(prefix_tokens)
        candidates = tuple(candidate_tokens)
        target_tokens = []

        for candidate in candidates:
            target_tokens.append(self.greedy_next(prefix))
            prefix.append(candidate)

        targets = tuple(target_tokens)
        match_length = 0
        for candidate, target in zip(candidates, targets):
            if candidate != target:
                break
            match_length += 1

        return VerificationResult(
            candidate_tokens=candidates,
            target_tokens=targets,
            greedy_match_length=match_length,
        )
