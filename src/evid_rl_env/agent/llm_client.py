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