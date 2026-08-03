"""Stable public facade for the reference Rejoin runtime."""

from typing import Optional, Sequence

from rejoin.config import DecoderConfig
from rejoin.models.protocols import DraftModel, TargetModel
from rejoin.runtime.direct_reattach import StepObserver, generate_rejoin
from rejoin.types import GenerationResult


class RejoinDecoder:
    """Configured exact-greedy decoder for one target/draft model pair."""

    def __init__(
        self,
        target: TargetModel,
        draft: DraftModel,
        config: Optional[DecoderConfig] = None,
    ) -> None:
        self._target = target
        self._draft = draft
        self._config = config or DecoderConfig()

    @property
    def target_model_id(self) -> str:
        return self._target.model_id

    @property
    def draft_model_id(self) -> str:
        return self._draft.model_id

    def generate(
        self,
        prefix_tokens: Sequence[int],
        max_new_tokens: int,
        observer: Optional[StepObserver] = None,
    ) -> GenerationResult:
        max_steps = self._config.max_steps or 1_000_000
        return generate_rejoin(
            target=self._target,
            draft=self._draft,
            prefix_tokens=prefix_tokens,
            max_new_tokens=max_new_tokens,
            draft_block_size=self._config.draft_block_size,
            min_suffix_length=self._config.min_suffix_length,
            stop_token_ids=self._config.stop_token_ids,
            max_steps=max_steps,
            observer=observer,
        )
