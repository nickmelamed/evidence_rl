from transformers import pipeline, logging 

logging.set_verbosity_error() # removing token warning

class LLMClient:
    def __init__(self, model_name="microsoft/phi-2", temperature=0.7):
        self.pipe = pipeline("text-generation", model=model_name)
        self.temperature = temperature

    def generate(self, prompt):
        out = self.pipe(
            prompt,
            max_new_tokens=128,
            do_sample=True,
            temperature=self.temperature,
            return_full_text=False
        )[0]["generated_text"]

        tokens = len(out.split())  # simple approximation
        return out, tokens

    def generate_structured(self, prompt, temperature=0.1):
        """Lower-temperature generation for structured outputs like JSON scores."""
        out = self.pipe(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=temperature,
            return_full_text=False
        )[0]["generated_text"]
        tokens = len(out.split())
        return out, tokens