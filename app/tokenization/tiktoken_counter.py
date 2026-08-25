import tiktoken

from app.tokenization.base import TokenCounter


class TiktokenCounter(TokenCounter):
    """
    Token counter backed by OpenAI's tiktoken library.

    For text-embedding-3-small, tiktoken currently maps the
    model to the cl100k_base encoding.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        try:
            # Use the model-to-encoding mapping maintained by
            # tiktoken instead of hard-coding cl100k_base here.
            self.encoding = tiktoken.encoding_for_model(
                model_name
            )
        except KeyError:
            raise ValueError(
                f"Unsupported tokenizer model: {model_name}"
            )

    def count(
        self,
        text: str,
    ) -> int:
        """
        Return the number of tokens in the text.
        """

        if not text:
            return 0

        return len(
            self.encoding.encode(text)
        )