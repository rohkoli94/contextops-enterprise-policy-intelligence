from abc import ABC, abstractmethod


class TokenCounter(ABC):
    """
    Abstraction for counting tokens.

    This keeps chunking and embedding logic independent
    of a specific tokenizer implementation.
    """

    @abstractmethod
    def count(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        raise NotImplementedError