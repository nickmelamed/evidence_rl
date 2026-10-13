"""
Unit tests for agent/llm_client.py's batched-sampling helpers
(generate_structured_n, _extract_texts), mocks the underlying
transformers pipeline directly (bypassing LLMClient.__init__ so no real
model ever loads) to verify the num_return_sequences>1 output-shape
handling, which isn't exercised by BestOfNBaseline's own tests (those
mock the client, not the pipeline extraction logic).
"""

from evid_rl_env.agent.llm_client import LLMClient, _extract_texts


def _make_client_with_fake_pipe(pipe_return):
    """LLMClient.__init__ loads a real transformers pipeline, bypass it
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


def test_derive_call_seed_depends_only_on_seed_prompt_and_occurrence():
    from evid_rl_env.agent.llm_client import derive_call_seed

    assert derive_call_seed(1, "p", 0) == derive_call_seed(1, "p", 0)
    assert derive_call_seed(1, "p", 0) != derive_call_seed(2, "p", 0)
    assert derive_call_seed(1, "p", 0) != derive_call_seed(1, "q", 0)
    assert derive_call_seed(1, "p", 0) != derive_call_seed(1, "p", 1)


def test_generation_seeds_torch_per_prompt_not_per_call_order(monkeypatch):
    import torch

    from evid_rl_env.agent import llm_client as module

    def draws(prompts):
        out = {}
        client = _make_client_with_fake_pipe([{"generated_text": "0"}])

        def fake_pipe(*args, **kwargs):
            out.setdefault(current[0], []).append(torch.rand(1).item())
            return [{"generated_text": "0"}]

        client._pipe = fake_pipe
        current = [None]
        for p in prompts:
            current[0] = p
            client.generate_structured(p)
        return out

    monkeypatch.setattr(module.torch.backends.mps, "is_available", lambda: False)
    alone = draws(["a"])
    with_other_calls_first = draws(["b", "c", "a"])
    assert alone["a"] == with_other_calls_first["a"]


def test_repeating_a_prompt_gives_a_new_sample(monkeypatch):
    import torch

    from evid_rl_env.agent import llm_client as module

    monkeypatch.setattr(module.torch.backends.mps, "is_available", lambda: False)
    client = _make_client_with_fake_pipe([{"generated_text": "0"}])
    values = []
    client._pipe = lambda *a, **k: (values.append(torch.rand(1).item()) or [{"generated_text": "0"}])
    client.generate_structured("same")
    client.generate_structured("same")
    assert values[0] != values[1]


class _FakeMessages:
    def __init__(self, text=None, exc=None):
        self._text, self._exc, self.calls = text, exc, 0

    def create(self, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        import types
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=self._text)])


def _annotator(text=None, exc=None):
    import types

    from evid_rl_env.agent.llm_client import AnnotatorClient

    client = AnnotatorClient.__new__(AnnotatorClient)
    client.model = "fake-model"
    client._cache = {}
    messages = _FakeMessages(text, exc)
    client._client = types.SimpleNamespace(messages=messages)
    return client, messages


def test_annotator_returns_the_parsed_index_and_caches_it():
    client, messages = _annotator(text="Action 4")
    assert client.select_action("obs", ["a"] * 13) == 4
    assert client.select_action("obs", ["a"] * 13) == 4
    assert messages.calls == 1


def test_annotator_clips_an_out_of_range_index():
    client, _ = _annotator(text="99")
    assert client.select_action("obs", ["a"] * 13) == 12


def test_annotator_raises_when_the_api_call_fails():
    import pytest

    from evid_rl_env.agent.llm_client import AnnotatorError

    client, _ = _annotator(exc=ConnectionError("down"))
    with pytest.raises(AnnotatorError, match="down"):
        client.select_action("obs", ["a"] * 13)


def test_annotator_raises_when_the_reply_has_no_index():
    import pytest

    from evid_rl_env.agent.llm_client import AnnotatorError

    client, messages = _annotator(text="no idea")
    with pytest.raises(AnnotatorError, match="could not parse"):
        client.select_action("obs", ["a"] * 13)
    assert client._cache == {}
