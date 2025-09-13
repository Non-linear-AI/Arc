"""OpenAI client wrapper for Arc functionality."""

import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam


class ArcTool:
    """Represents a tool that can be called by the AI."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ArcToolCall:
    """Represents a tool call from the AI."""

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.name = name
        self.arguments = arguments

    @classmethod
    def from_openai_tool_call(cls, tool_call: Any) -> "ArcToolCall":
        """Create from OpenAI tool call format."""
        return cls(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )


class ArcClient:
    """OpenAI API client wrapper with Arc-specific functionality."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: str | None = None,
    ):
        self.current_model = model
        self.base_url = base_url or os.getenv(
            "ARC_BASE_URL", "https://api.openai.com/v1"
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(360.0),
        )

    def set_model(self, model: str) -> None:
        """Update the current model."""
        self.current_model = model

    def get_current_model(self) -> str:
        """Get the current model."""
        return self.current_model

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ArcTool] | None = None,
        model: str | None = None,
    ) -> ChatCompletionMessage:
        """Send a chat completion request."""
        try:
            request_payload = {
                "model": model or self.current_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4000,
            }

            if tools:
                request_payload["tools"] = [tool.to_dict() for tool in tools]
                request_payload["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**request_payload)

            return response.choices[0].message

        except Exception as e:
            raise Exception(f"Arc API error: {str(e)}") from e

    async def chat_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ArcTool] | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Send a streaming chat completion request."""
        try:
            request_payload = {
                "model": model or self.current_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4000,
                "stream": True,
            }

            if tools:
                request_payload["tools"] = [tool.to_dict() for tool in tools]
                request_payload["tool_choice"] = "auto"

            stream = await self.client.chat.completions.create(**request_payload)

            async for chunk in stream:
                yield chunk

        except Exception as e:
            raise Exception(f"Arc API error: {str(e)}") from e
