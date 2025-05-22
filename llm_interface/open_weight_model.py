from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llm_interface.base import LLMInterface
import os

class OpenWeightModel(LLMInterface):
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        access_token = os.getenv("HUGGINGFACE_TOKEN")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token).cpu()

    def query(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output[len(prompt):].strip()
