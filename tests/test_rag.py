from query_data import query_rag
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json


def load_eval_prompt():
    with open("prompts.json", "r") as f:
        prompts = json.load(f)
        return prompts["eval_prompt"]["template"]

def query_and_validate(question: str, expected_response: str) -> bool:
    actual_response = query_rag(question)
    prompt_string = load_eval_prompt()
    prompt_template = PromptTemplate.from_template(prompt_string)
    prompt = prompt_template.format(
        expected_response=expected_response, 
        actual_response=actual_response
    )
    
    evaluator_model = OllamaLLM(model="llama3.2")
    evaluation_result = evaluator_model.invoke(prompt).strip().lower()
    
    print(f"\n[Evaluator Decision]: {evaluation_result}")
    
    if "true" in evaluation_result:
        return True
    elif "false" in evaluation_result:
        return False
    else:
        raise ValueError(f"Evaluator returned an unexpected format: {evaluation_result}")

def test_oop_rules():
    expected = (
        "Multi-head attention is a mechanism that runs multiple attention layers "
        "(or heads) in parallel. This allows the model to jointly attend to "
        "information from different representation subspaces at different positions."
    )
    
    assert query_and_validate(
        question="What is multihead attention?",
        expected_response=expected
    )