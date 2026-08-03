"""Demonstrate a local draft error whose suffix is recovered exactly."""

from typing import Sequence, Tuple

from rejoin import DecoderConfig, RejoinDecoder
from rejoin.models.base import ReferenceTargetModel


class PositionTarget(ReferenceTargetModel):
    def __init__(self, continuation: Sequence[int]) -> None:
        self._continuation = tuple(continuation)

    @property
    def model_id(self) -> str:
        return "demo-target"

    def greedy_next(self, prefix_tokens: Sequence[int]) -> int:
        return self._continuation[len(prefix_tokens)]


class OneMistakeDraft:
    @property
    def model_id(self) -> str:
        return "demo-draft"

    def generate_greedy(
        self, prefix_tokens: Sequence[int], max_new_tokens: int
    ) -> Tuple[int, ...]:
        continuation = [10, 20, 30, 40, 50]
        start = len(prefix_tokens)
        proposal = continuation[start : start + max_new_tokens]
        if start == 0 and len(proposal) > 1:
            proposal[1] = 999
        return tuple(proposal)


def main() -> None:
    decoder = RejoinDecoder(
        target=PositionTarget([10, 20, 30, 40, 50]),
        draft=OneMistakeDraft(),
        config=DecoderConfig(draft_block_size=5, min_suffix_length=1),
    )
    result = decoder.generate([], max_new_tokens=5)
    print(f"tokens: {result.tokens}")
    print(f"reused suffix tokens: {result.reused_suffix_tokens}")
    print(f"target invocations: {result.target_invocations}")


if __name__ == "__main__":
    main()
