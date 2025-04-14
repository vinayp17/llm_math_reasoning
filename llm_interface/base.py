from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def query(self, prompt: str) -> str:
        """Send a prompt and return the LLM's response."""
        pass
