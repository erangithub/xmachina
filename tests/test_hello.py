from xmachina import build_context
from xmachina.llms import EchoLLM
from xmachina.environment import Environment

def test_hello_world_replay():
    env = Environment(llm=EchoLLM(), input_fn=input, continue_live=True)
    env.add_message("user", "hello")
    env.add_message("assistant", "echo: hello")
    env.rewind()

    env.input()  # replays user message
    response = env.llm_complete(build_context(env.history()))
    
    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_full_replay passed")


def test_hello_world_replay_then_live():
    env = Environment(llm=EchoLLM(), input_fn=input, continue_live=True)
    env.add_message("user", "hello")
    env.rewind()
   
    env.input()  # replays user message
    response = env.llm_complete(build_context(env.history()))

    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_replay_then_live passed")


def test_hello_world_live():
    env = Environment(llm=EchoLLM(), input_fn=lambda: "hello")
    
    env.input()  # reads user message
    response = env.llm_complete(build_context(env.history()))

    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_live passed")


if __name__ == "__main__":
    test_hello_world_replay()
    test_hello_world_replay_then_live()
    test_hello_world_live()
