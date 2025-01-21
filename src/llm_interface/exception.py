class LlmException(Exception):
    """Base class for LLM exceptions."""

    pass


class RateLimitException(LlmException):
    """LLM API rate limit exceeded."""

    pass
