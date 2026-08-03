"""Suffix-reuse metrics."""


def suffix_salvage_ratio(reused_suffix_tokens: int, available_suffix_tokens: int) -> float:
    if reused_suffix_tokens < 0 or available_suffix_tokens < 0:
        raise ValueError("token counts cannot be negative")
    if reused_suffix_tokens > available_suffix_tokens:
        raise ValueError("reused suffix cannot exceed the available suffix")
    if available_suffix_tokens == 0:
        return 0.0
    return reused_suffix_tokens / available_suffix_tokens
