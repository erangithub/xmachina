"""
noagent.mock — development and testing utilities

These are not production components. They are the simplest possible
implementations of the LLM interface, useful for unit tests and examples
without any external dependencies.
"""
import json
from noagent import Message, ToolCall


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class EchoLLM:
    """Echoes the last user message. The simplest possible LLM."""

    def complete(self, context: list[Message]) -> Message:
        last_user = next(m for m in reversed(context) if m.role == "user")
        return Message("assistant", f"echo: {last_user.content}")


class ToolCallLLM:
    """
    Simulates a two-turn tool use interaction:
      turn 1 — returns a tool call
      turn 2 — returns a final answer once a tool result is in context

    Useful for testing tool use loops without a real LLM.
    """

    def __init__(self, tool_name: str, arguments: dict, final_answer: str):
        self.tool_name    = tool_name
        self.arguments    = arguments
        self.final_answer = final_answer
        self._call_id     = "call_001"

    def complete(self, context: list[Message]) -> Message:
        if any(m.role == "tool" for m in context):
            return Message("assistant", self.final_answer)
        return Message(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(
                    id=self._call_id,
                    name=self.tool_name,
                    arguments=json.dumps(self.arguments),
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Mock tools
# ---------------------------------------------------------------------------

def get_weather(location: str) -> str:
    """Returns a fixed weather string. Stand-in for a real weather API."""
    return f"25c and sunny in {location}"


# Hand-written OpenAI-compatible schema for get_weather.
# In production generate this with the OpenAI SDK, Pydantic, FastMCP, etc.
tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": get_weather.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }
]