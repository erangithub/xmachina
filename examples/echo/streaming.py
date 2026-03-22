from xmachina import Message, build_context
from xmachina.llms import EchoLLM
from xmachina.environment import Environment


def main():
    env = Environment(llm=EchoLLM(), continue_live=True, input_fn=input)
    env.add_message(Message("user", "And I think to myself, what a wonderful world."))
    #env.add_message(Message("assistant", "echo: And I think to myself, what a wonderful world."))

    env.rewind()

    env.input()
    full_content = ""
    for delta in env.llm_stream(build_context(env.history())):
        print(delta.content, end="", flush=True)
        full_content += delta.content

    print(f"\nLogged {len(list(env.history().iter_messages()))} messages")


if __name__ == "__main__":
    main()
