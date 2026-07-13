"""
Unit tests for agent/llm_client.py's batched-sampling helpers
(generate_structured_n, _extract_texts) — mocks the underlying
transformers pipeline directly (bypassing LLMClient.__init__ so no real
model ever loads) to verify the num_return_sequences>1 output-shape
handling, which isn't exercised by BestOfNBaseline's own tests (those
mock the client, not the pipeline extraction logic).
"""

from evid_rl_env.agent.llm_client import LLMClient, _extract_texts


def _make_client_with_fake_pipe(pipe_return):
    """LLMClient.__init__ loads a real transformers pipeline — bypass it
    entirely via object.__new__ and set _pipe to a plain callable."""
    client = object.__new__(LLMClient)
    client.model_name = "fake-model"
    client.temperature = 0.7
    client.seed = 42
    client._pipe = lambda *args, **kwargs: pipe_return
    return client


def test_extract_texts_handles_chat_template_format():
    output = [
        {"generated_text": [{"role": "assistant", "content": "0"}]},
        {"generated_text": [{"role": "assistant", "content": "1"}]},
        {"generated_text": [{"role": "assistant", "content": "2"}]},
    ]
    assert _extract_texts(output) == ["0", "1", "2"]


def test_extract_texts_handles_plain_string_format():
    output = [{"generated_text": "0"}, {"generated_text": "1"}]
    assert _extract_texts(output) == ["0", "1"]


def test_generate_structured_n_returns_n_text_token_pairs():
    fake_output = [
        {"generated_text": [{"role": "assistant", "content": "0"}]},
        {"generated_text": [{"role": "assistant", "content": "1"}]},
        {"generated_text": [{"role": "assistant", "content": "2"}]},
    ]
    client = _make_client_with_fake_pipe(fake_output)

    results = client.generate_structured_n("some prompt", n=3)

    assert results == [("0", 1), ("1", 1), ("2", 1)]


def test_generate_structured_n_passes_num_return_sequences(monkeypatch):
    captured = {}

    def _fake_pipe(chat, **kwargs):
        captured.update(kwargs)
        return [{"generated_text": [{"role": "assistant", "content": "0"}]}] * kwargs["num_return_sequences"]

    client = object.__new__(LLMClient)
    client.model_name = "fake-model"
    client.temperature = 0.7
    client.seed = 42
    client._pipe = _fake_pipe

    client.generate_structured_n("prompt", n=5, temperature=0.2)

    assert captured["num_return_sequences"] == 5
    assert captured["temperature"] == 0.2
    assert captured["max_new_tokens"] == 32
