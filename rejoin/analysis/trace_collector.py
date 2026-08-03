"""Thread-safe JSONL trace collection for offline experiments."""

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from rejoin.types import RejoinStepResult


@dataclass(frozen=True)
class StepTrace:
    schema_version: int
    prompt_id: str
    task_type: str
    target_model: str
    draft_model: str
    accepted_prefix_length: int
    draft_length: int
    first_mismatch_position: Optional[int]
    suffix_length: int
    direct_survival_length: int
    attempted_rejoin: bool
    tokens_committed: int
    target_invocations: int
    metadata: Mapping[str, Any]

    @classmethod
    def from_step(
        cls,
        result: RejoinStepResult,
        *,
        prompt_id: str,
        task_type: str,
        target_model: str,
        draft_model: str,
        accepted_prefix_length: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StepTrace":
        initial = result.initial_verification
        mismatch = None if initial.fully_matches else initial.greedy_match_length
        suffix_length = 0 if mismatch is None else len(result.draft_tokens) - mismatch - 1
        survival = result.reused_suffix_tokens
        return cls(
            schema_version=1,
            prompt_id=prompt_id,
            task_type=task_type,
            target_model=target_model,
            draft_model=draft_model,
            accepted_prefix_length=accepted_prefix_length,
            draft_length=len(result.draft_tokens),
            first_mismatch_position=mismatch,
            suffix_length=suffix_length,
            direct_survival_length=survival,
            attempted_rejoin=result.attempted_rejoin,
            tokens_committed=len(result.committed_tokens),
            target_invocations=result.target_invocations,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JsonlTraceCollector:
    """Append one durable, independently parseable trace per line."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def append(self, trace: StepTrace) -> None:
        line = json.dumps(trace.to_dict(), separators=(",", ":"), sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
