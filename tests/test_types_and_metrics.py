import unittest

from rejoin.metrics.exactness import assert_exact_match
from rejoin.metrics.salvage import suffix_salvage_ratio
from rejoin.types import VerificationResult


class TypesAndMetricsTests(unittest.TestCase):
    def test_verification_rejects_incorrect_match_length(self) -> None:
        with self.assertRaises(ValueError):
            VerificationResult(
                candidate_tokens=(1, 9),
                target_tokens=(1, 2),
                greedy_match_length=2,
            )

    def test_exactness_error_identifies_first_divergence(self) -> None:
        with self.assertRaisesRegex(AssertionError, "position 1"):
            assert_exact_match([1, 9, 3], [1, 2, 3])

    def test_salvage_ratio_validates_counts(self) -> None:
        self.assertEqual(suffix_salvage_ratio(3, 4), 0.75)
        self.assertEqual(suffix_salvage_ratio(0, 0), 0.0)
        with self.assertRaises(ValueError):
            suffix_salvage_ratio(5, 4)


if __name__ == "__main__":
    unittest.main()
