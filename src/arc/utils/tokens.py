"""Token counting utilities."""

import tiktoken
from openai.types.chat import ChatCompletionMessageParam


class TokenCounter:
    """Token counting utility."""

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def count_message_tokens(self, messages: list[ChatCompletionMessageParam]) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for message in messages:
            # Count tokens in content
            if isinstance(message.get("content"), str):
                total += self.count_tokens(message["content"])

            # Count tokens in role
            total += self.count_tokens(message["role"])

            # Add tokens for message structure
            total += 4  # Every message has some overhead

        # Add tokens for assistant reply start
        total += 2

        return total

    def estimate_streaming_tokens(self, content: str) -> int:
        """Estimate tokens in streaming content."""
        return self.count_tokens(content)
