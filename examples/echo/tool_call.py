import json

from xmachina import Message, EventLog, build_context
from xmachina.mock import ToolCallLLM, get_weather, tool_schemas


def main():
    tools = {"get_weather": get_weather}

    llm = ToolCallLLM(
        tool_name="get_weather",
        arguments={"location": "london"},
        final_answer="The weather in London is 25c and sunny.",
    )
    tool_injections = [json.dumps(s) for s in tool_schemas]

    conversation = EventLog.start(
        Message("user", "what's the weather in london?")
    )

    while True:
        # build_context decides what the LLM sees — ephemeral, discarded after each call
        # conversation.append records what happened — permanent, never modified
        # these are two separate concerns, on two separate objects, always explicit
        context  = build_context(conversation, injections=tool_injections)
        response = llm.complete(context)
        conversation = conversation.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                # routing is a dict lookup — nothing more
                fn     = tools[tool_call.name]
                args   = json.loads(tool_call.arguments)
                result = str(fn(**args))
                conversation = conversation.append(
                    Message("tool", result, tool_call_id=tool_call.id)
                )
        else:
            break

    for msg in conversation.messages():
        print(f"[{msg.role}] {msg.content or msg.tool_calls}")


if __name__ == "__main__":
    main()