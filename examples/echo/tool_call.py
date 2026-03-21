import json

from xmachina import Message, EventLog, build_context, ToolCall
from xmachina.mock import ToolCallLLM, get_weather, tool_schemas
from xmachina.environment import replay, Tool


def main():
    def get_weather_tool(location: str) -> str:
        return f"25c and sunny in {location}"

    tools = [
        Tool(name="get_weather", fn=get_weather_tool, schema=tool_schemas[0])
    ]

    llm = ToolCallLLM(
        tool_name="get_weather",
        arguments={"location": "london"},
        final_answer="The weather in London is 25c and sunny.",
    )

    log = EventLog()
    log = log.append(Message("user", "what's the weather in london?"))
    log = log.append(Message("assistant", content=None, tool_calls=(ToolCall(id="1", name="get_weather", arguments='{"location": "london"}'),)))
    log = log.append(Message("tool", "25c and sunny in london", tool_call_id="1"))
    log = log.append(Message("assistant", "The weather in London is 25c and sunny."))
    env = replay(log, llm=llm, tools=tools)

    env.input()  # replays user message
    while True:
        context = build_context(env.log, injections=[json.dumps(s) for s in tool_schemas])
        response = env.llm_complete(context)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                env.call_tool(tool_call)
        else:
            break

    for msg in env.log.messages():
        print(f"[{msg.role}] {msg.content or msg.tool_calls}")


if __name__ == "__main__":
    main()