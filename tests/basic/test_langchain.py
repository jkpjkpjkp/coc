
import pytest
from langchain_openai import OpenAI

def test_simple_invoke():
    llm = OpenAI()
    result = llm.invoke("hi!")
    assert isinstance(result, str)
    assert "h" in result.lower()


def test_basics():
    llm = OpenAI()
    assert 'gpt' in llm.model_name.lower()