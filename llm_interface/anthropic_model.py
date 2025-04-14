import anthropic
from llm_interface.base import LLMInterface

class AnthropicModel(LLMInterface):
    def __init__(self, model_name: str, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def query(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
