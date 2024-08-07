from anthropic import Anthropic
from openai import OpenAI
from os import getenv
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict

class Engine(ABC):
    @abstractmethod
    def get_response(self, messages: Union[str, List[Dict[str, str]]], system_prompt: str = "") -> str:
        pass

class ClaudeModel(Engine):
    def __init__(self):
        self.client = Anthropic(api_key=getenv("ANTHROPIC_API_KEY"))

    def get_response(self, messages: Union[str, List[Dict[str, str]]], system_prompt: str = "") -> str:
        """ 
        Obtain response from Claude
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content

class OpenAIModel(Engine):
    def __init__(self):
        self.client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

    def get_response(self, messages: Union[str, List[Dict[str, str]]], system_prompt: str = "") -> str:
        """
        Obtain response from OpenAI
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
        )
        return response.choices[0].message.content

# Initialize the models
def route_model(model_name: str) -> Engine:
    assert model_name in ["OpenAI", "Claude"], "Model not supported"
    if model_name == "OpenAI":
        return OpenAIModel()
    else:
        return ClaudeModel()