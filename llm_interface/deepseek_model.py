import requests
from llm_interface.base import LLMInterface

class DeepSeekModel(LLMInterface):
    def __init__(self, model_name: str, api_key: str, api_url: str):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url  # e.g., https://api.deepseek.com/v1

    def query(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, #recommended to use 0 for math/coding
        }

        response = requests.post(self.api_url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
