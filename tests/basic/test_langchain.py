
import pytest
from langchain_openai import OpenAI

def test_basics():
    llm = OpenAI()
    assert 'gpt' in llm.model_name.lower()