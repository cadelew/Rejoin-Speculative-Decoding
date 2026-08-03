import unittest
from typing import Sequence

from rejoin.models.base import ReferenceTargetModel
from rejoin.runtime.autoregressive import generate_target_greedy
from rejoin.runtime.direct_reattach import generate_rejoin, rejoin_exact_step
from rejoin.runtime.speculative import generate_speculative, speculative_step
from tests.fakes import FunctionDraft, PositionTarget


class PrefixSensitiveTarget(ReferenceTargetModel):
    @property
    def model_id(self) -> str:
        return "prefix-sensitive-target"

    def greedy_next(self, prefix_tokens: Sequence[int]) -> int:
        weighted_prefix = sum(
            (index + 1) * token for index, token in enumerate(prefix_tokens)
        )
        return (weighted_prefix + 17 * len(prefix_tokens) + 11) % 101 + 1


class RuntimeTests(unittest.TestCase):
    def test_direct_reattachment_recovers_local_error_suffix(self) -> None:
        target = PositionTarget([1, 2, 3, 4])
        draft = FunctionDraft(lambda _prefix, _limit: [1, 99, 3, 4])

        result = rejoin_exact_step(
            target=target,
            draft=draft,
            prefix_tokens=[],
            draft_length=4,
            min_suffix_length=1,
        )

        self.assertEqual(result.committed_tokens, (1, 2, 3, 4))
        self.assertTrue(result.attempted_rejoin)
        self.assertEqual(result.reused_suffix_tokens, 2)
        self.assertEqual(result.target_invocations, 2)

    def test_short_suffix_uses_ordinary_fallback(self) -> None:
        target = PositionTarget([1, 2, 3])
        draft = FunctionDraft(lambda _prefix, _limit: [1, 99, 3])

        result = rejoin_exact_step(
            target=target,
            draft=draft,
            prefix_tokens=[],
            draft_length=3,
            min_suffix_length=2,
        )

        self.assertEqual(result.committed_tokens, (1, 2))
        self.assertFalse(result.attempted_rejoin)
        self.assertEqual(result.target_invocations, 1)

    def test_speculative_step_commits_correction_at_first_mismatch(self) -> None:
        target = PositionTarget([7, 8, 9])
        draft = FunctionDraft(lambda _prefix, _limit: [7, 88, 9])

        committed, _, verification = speculative_step(target, draft, [], 3)

        self.assertEqual(committed, (7, 8))
        self.assertEqual(verification.greedy_match_length, 1)

    def test_all_runtimes_match_target_greedy(self) -> None:
        continuation = [10, 20, 30, 40, 50, 60, 70]
        target = PositionTarget(continuation)

        def propose(prefix, limit):
            start = len(prefix)
            tokens = continuation[start : start + limit]
            if start == 0 and len(tokens) > 1:
                tokens = list(tokens)
                tokens[1] = 999
            return tokens

        draft = FunctionDraft(propose)
        expected = generate_target_greedy(target, [], len(continuation)).tokens
        speculative = generate_speculative(
            target, draft, [], len(continuation), draft_block_size=4
        )
        rejoin = generate_rejoin(
            target,
            draft,
            [],
            len(continuation),
            draft_block_size=4,
            min_suffix_length=1,
        )

        self.assertEqual(speculative.tokens, expected)
        self.assertEqual(rejoin.tokens, expected)
        self.assertGreater(rejoin.reused_suffix_tokens, 0)

    def test_rejected_suffix_is_never_trusted_under_prefix_shift(self) -> None:
        target = PrefixSensitiveTarget()
        draft = FunctionDraft(lambda _prefix, limit: [0] * limit)
        expected = generate_target_greedy(target, [], 25)

        result = generate_rejoin(
            target,
            draft,
            [],
            max_new_tokens=25,
            draft_block_size=5,
            min_suffix_length=1,
        )

        self.assertEqual(result.tokens, expected.tokens)
        self.assertEqual(result.reused_suffix_tokens, 0)

    def test_stop_token_truncates_committed_block(self) -> None:
        target = PositionTarget([1, 2, 3, 4])
        draft = FunctionDraft(lambda prefix, limit: [1, 2, 3, 4][len(prefix) :][:limit])

        result = generate_rejoin(
            target,
            draft,
            [],
            max_new_tokens=4,
            draft_block_size=4,
            min_suffix_length=1,
            stop_token_ids={2},
        )

        self.assertEqual(result.tokens, (1, 2))

    def test_stop_before_retained_suffix_does_not_count_reuse(self) -> None:
        target = PositionTarget([1, 2, 3, 4])
        draft = FunctionDraft(lambda _prefix, _limit: [1, 99, 3, 4])

        result = generate_rejoin(
            target,
            draft,
            [],
            max_new_tokens=4,
            draft_block_size=4,
            min_suffix_length=1,
            stop_token_ids={2},
        )

        self.assertEqual(result.tokens, (1, 2))
        self.assertEqual(result.reused_suffix_tokens, 0)

    def test_zero_requested_tokens_does_not_call_models(self) -> None:
        target = PositionTarget([])
        draft = FunctionDraft(lambda _prefix, _limit: self.fail("draft should not run"))

        result = generate_rejoin(target, draft, [], 0, draft_block_size=4)

        self.assertEqual(result.tokens, ())
        self.assertEqual(result.steps, 0)


if __name__ == "__main__":
    unittest.main()
