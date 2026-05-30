import logging as _logging
import re
import random

from transformers import pipeline, logging

logging.set_verbosity_error()

_logger = _logging.getLogger(__name__)


def _extract_text(output):
    """Safely extract generated text from pipeline output regardless of format."""
    result = output[0]["generated_text"]
    if isinstance(result, list):
        # chat template format — last message is assistant turn
        return result[-1].get("content", "")
    return result


class LLMClient:
    """
    Instruction-tuned generation model for producing arguments, summaries,
    and other natural language outputs during episodes.

    Default: google/gemma-2-2b-it
    """

    def __init__(self, model_name="google/gemma-2-2b-it", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
        )

    @property
    def pipe(self):
        return self._pipe

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
            max_new_tokens=256,
            do_sample=True,
            temperature=temperature,
            return_full_text=False,
        )
        text = _extract_text(out)
        tokens = len(text.split())
        return text, tokens


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
    ):
        self.model_name = model_name
        self.temperature = temperature
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
        )

    def _chat(self, prompt):
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt):
        out = self._pipe(
            self._chat(prompt),
            max_new_tokens=256,
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

    def select_action(self, observation: str, action_descriptions: list) -> int:
        """
        Ask the model to choose the best action index for the given observation.
        Falls back to a random valid index on parse failure.
        """
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
                return int(max(0, min(int(match.group()), len(action_descriptions) - 1)))
        except Exception as exc:
            _logger.warning("AnnotatorClient: LLM call failed (%s), returning random.", exc)
        return random.randint(0, len(action_descriptions) - 1)
