import json
import tempfile
import unittest
from pathlib import Path

from rejoin.analysis.rejoinability import measure_direct_rejoinability
from rejoin.analysis.trace_collector import JsonlTraceCollector, StepTrace
from rejoin.runtime.direct_reattach import rejoin_exact_step
from tests.fakes import FunctionDraft, PositionTarget


class AnalysisAndTraceTests(unittest.TestCase):
    def test_measure_direct_rejoinability(self) -> None:
        target = PositionTarget([1, 2, 3, 4, 5])
        draft = FunctionDraft(lambda _prefix, _limit: [1, 99, 3, 4, 5])

        result = measure_direct_rejoinability(target, draft, [], 5)

        self.assertFalse(result.fully_accepted)
        self.assertEqual(result.mismatch_index, 1)
        self.assertEqual(result.suffix_length, 3)
        self.assertEqual(result.direct_survival_length, 3)
        self.assertEqual(result.suffix_salvage_ratio, 1.0)

    def test_jsonl_trace_is_versioned_and_parseable(self) -> None:
        target = PositionTarget([1, 2, 3, 4])
        draft = FunctionDraft(lambda _prefix, _limit: [1, 99, 3, 4])
        step = rejoin_exact_step(target, draft, [], 4, min_suffix_length=1)
        trace = StepTrace.from_step(
            step,
            prompt_id="example-1",
            task_type="unit_test",
            target_model=target.model_id,
            draft_model=draft.model_id,
            accepted_prefix_length=0,
            metadata={"split": "test"},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "traces.jsonl"
            JsonlTraceCollector(path).append(trace)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["direct_survival_length"], 2)
        self.assertEqual(payload["metadata"], {"split": "test"})


if __name__ == "__main__":
    unittest.main()
