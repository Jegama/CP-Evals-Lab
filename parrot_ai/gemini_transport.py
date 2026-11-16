"""Shared, bounded network recovery for Gemini generation and judging."""

import os


def gemini_http_options():
    from google.genai import types

    timeout_seconds = int(os.environ.get("CP_GEMINI_TIMEOUT_SECONDS", "120"))
    attempts = int(os.environ.get("CP_GEMINI_MAX_ATTEMPTS", "5"))
    if timeout_seconds < 1 or attempts < 1:
        raise ValueError("Gemini timeout and maximum attempts must be positive")
    return types.HttpOptions(
        api_version="v1",
        timeout=timeout_seconds * 1000,
        retry_options=types.HttpRetryOptions(
            attempts=attempts,
            initial_delay=5,
            max_delay=30,
            exp_base=2,
            jitter=1,
        ),
    )
