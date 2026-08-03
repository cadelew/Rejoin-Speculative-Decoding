"""Output-equivalence checks."""

from typing import Sequence


def assert_exact_match(actual: Sequence[int], expected: Sequence[int]) -> None:
    """Raise an informative assertion at the earliest output divergence."""

    for index, (actual_token, expected_token) in enumerate(zip(actual, expected)):
        if actual_token != expected_token:
            raise AssertionError(
                f"token mismatch at position {index}: "
                f"actual={actual_token}, expected={expected_token}"
            )
    if len(actual) != len(expected):
        raise AssertionError(
            f"output lengths differ: actual={len(actual)}, expected={len(expected)}"
        )
