import logging as _logging
import os
import random
import re

# Disable the fast tokenizer's Rust-backed parallel worker pool. Without this,
# the pool's semaphores leak at interpreter shutdown and cause a segfault.
# Use an unconditional assignment so a shell-level TOKENIZERS_PARALLELISM=true
# cannot override this and re-introduce the leak.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import logging, pipeline
from transformers import set_seed as _transformers_set_seed

logging.set_verbosity_error()

_logger = _logging.getLogger(__name__)

# Use MPS on Apple Silicon if available, otherwise CPU.
# set_seed (which triggers torch.mps.manual_seed) is called once at __init__,
# not before every inference call, so MPS state is not repeatedly reset.
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _extract_text(output):
    """Safely extract generated text from pipeline output regardless of format."""
    result = output[0]["generated_text"]
    if isinstance(result, list):
        # chat template format — last message is assistant turn
        return result[-1].get("content", "")
    return result


def _extract_texts(output):
    """Same as _extract_text, but for a num_return_sequences>1 pipeline
    call — output is a list of one generated_text entry per sequence
    (a single input prompt, not a batch of different prompts)."""
    texts = []
    for item in output:
        result = item["generated_text"]
        if isinstance(result, list):
            texts.append(result[-1].get("content", ""))
        else:
            texts.append(result)
    return texts


class LLMClient:
    """
    Instruction-tuned generation model for producing arguments, summaries,
    and other natural language outputs during episodes.

    Default: google/gemma-2-2b-it
    """

    def __init__(self, model_name="google/gemma-2-2b-it", temperature=0.7, seed: int = 42):
        self.model_name = model_name
        self.temperature = temperature
        # store seed so every pipeline call sets it via transformers.set_seed,
        self.seed = seed
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device=_DEVICE,
        )
        # Seed once at init
        _transformers_set_seed(self.seed)

    def _chat(self, prompt):
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt):
        out = self._pipe(
            self._chat(prompt),
            max_new_tokens=128,
            do_sample=True,
            temperature=self.temperature,
            return_full_text=False,
        )
        text = _extract_text(out)
        tokens = len(text.split())
        return text, tokens

    def generate_structured(self, prompt, temperature=0.1):
        """Lower temperature for structured outputs like JSON."""
        out = self._pipe(
            self._chat(prompt),
            max_new_tokens=32,   # callers only need an action index (1-2 tokens)
            do_sample=True,
            temperature=temperature,
            return_full_text=False,
        )
        text = _extract_text(out)
        tokens = len(text.split())
        return text, tokens

    def generate_structured_n(self, prompt, n: int, temperature=0.1):
        """Batched sibling of generate_structured: samples n completions of
        the *same* prompt in one forward pass (num_return_sequences=n)
        instead of n sequential pipeline calls — used by BestOfNBaseline,
        which otherwise called generate_structured n times for a literally
        identical input."""
        out = self._pipe(
            self._chat(prompt),
            max_new_tokens=32,
            do_sample=True,
            temperature=temperature,
            return_full_text=False,
            num_return_sequences=n,
        )
        texts = _extract_texts(out)
        return [(text, len(text.split())) for text in texts]


class JudgeLLMClient:
    """
    Separate instruction-tuned model dedicated to structured JSON scoring.
    Kept separate from the actor model to avoid conflicts and allow
    independent temperature control.

    Default: Qwen/Qwen2.5-1.5B-Instruct
    Can use a smaller model (Qwen2.5-0.5B-Instruct) if memory constrained.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        temperature=0.1,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device=_DEVICE,
        )
        _transformers_set_seed(self.seed)

    def _chat(self, prompt):
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt):
        out = self._pipe(
            self._chat(prompt),
            max_new_tokens=128,  
            do_sample=True,
            temperature=self.temperature,
            return_full_text=False,
        )
        text = _extract_text(out)
        tokens = len(text.split())
        return text, tokens

    # alias so LLMJudge works with either client
    def generate_structured(self, prompt, temperature=None):
        return self.generate(prompt)


class AnnotatorClient:
    """
    Strong LLM annotator for labeling trajectories via the Anthropic Messages API.

    Default: claude-opus-4-5
    Requires ANTHROPIC_API_KEY in the environment (loaded via python-dotenv).
    """

    def __init__(self, model: str = "claude-opus-4-5"):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required for AnnotatorClient. "
                "Install with: pip install anthropic"
            ) from exc

        from dotenv import load_dotenv
        load_dotenv()

        self.model = model
        self._client = anthropic.Anthropic()
        # cache API responses keyed on (observation, actions) so identical
        # prompts within a collection run don't generate redundant billable API calls;

        self._cache: dict = {}

    def select_action(self, observation: str, action_descriptions: list) -> int:
        """
        Ask the model to choose the best action index for the given observation.
        Falls back to a random valid index on parse failure.
        """
        # check per-instance cache before making an API call

        cache_key = (observation, tuple(action_descriptions))
        if cache_key in self._cache:
            return self._cache[cache_key]

        desc = "\n".join(
            f"{i}: {a}" for i, a in enumerate(action_descriptions)
        )
        prompt = (
            "You are selecting the best next action for an evidence retrieval task.\n"
            f"Observation: {observation}\n"
            f"Available actions:\n{desc}\n"
            "Respond with only the index number of the best action. No explanation."
        )
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            match = re.search(r"\d+", text)
            if match:
                result = int(max(0, min(int(match.group()), len(action_descriptions) - 1)))
                self._cache[cache_key] = result
                return result
            # include model and observation context so parse failures are debuggable
            _logger.warning(
                "AnnotatorClient: could not parse index from response %r | model=%s | obs='%.80s'",
                text, self.model, observation,
            )
        except Exception as exc:
            _logger.warning(
                "AnnotatorClient: LLM call failed (%s) | model=%s | obs='%.80s' — returning random.",
                exc, self.model, observation,
            )
        return random.randint(0, len(action_descriptions) - 1)
