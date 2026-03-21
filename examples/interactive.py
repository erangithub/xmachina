from xmachina import Message, EventLog, build_context
from xmachina.llms import LMStudioLLM
from xmachina.environment import live


def main():
    def get_input():
        return input("You: ")

    env = live(llm=LMStudioLLM(), input_fn=get_input)

    while True:
        user_input = env.input()
        if user_input.lower() in ("quit", "exit"):
            break
        response = env.llm_complete(build_context(env.log))
        print(f"Assistant: {response.content}")

if __name__ == "__main__":
    main()
