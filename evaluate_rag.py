import os
import sys
import random
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from get_embedding_function import get_embedding_function
from query_data import query_rag
from ragas.run_config import RunConfig
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _is_ci() -> bool:
    return os.getenv("GITHUB_ACTIONS") == "true"


def _apply_smoke_sampling(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Use a small random subset in CI to keep PR checks fast and inexpensive."""
    if not _is_ci():
        return df, False

    min_n = int(os.getenv("SMOKE_TEST_MIN_QUESTIONS", "3"))
    max_n = int(os.getenv("SMOKE_TEST_MAX_QUESTIONS", "5"))
    if min_n > max_n:
        min_n, max_n = max_n, min_n

    sample_n = min(len(df), random.randint(min_n, max_n))
    sampled_df = df.sample(n=sample_n)
    print(f"Smoke test mode enabled (CI): evaluating {sample_n}/{len(df)} questions.")
    return sampled_df, True

def run_evaluations():
    df = pd.read_csv("candidate_dataset.csv")
    df, smoke_test_mode = _apply_smoke_sampling(df)
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for index, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        expected_context = eval(row["contexts"]) 
        
        response = query_rag(question) 

        if isinstance(response, str) and response.startswith("LLM generation unavailable"):
            print(f"Evaluation aborted: {response}")
            sys.exit(1)
        
        data["question"].append(question)
        data["answer"].append(response)
        data["contexts"].append(expected_context)
        data["ground_truth"].append(ground_truth)

    dataset = Dataset.from_dict(data)

    if os.getenv("GITHUB_ACTIONS") == "true":
        raw_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0,
            api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        raw_llm = ChatOllama(
            model="llama3.1", 
            temperature=0, 
            format="json"
        )

    raw_embeddings = get_embedding_function()
    
    evaluator_llm = LangchainLLMWrapper(raw_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    safe_config = RunConfig(max_workers=1, timeout=240)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=safe_config
    )

    print(result)
    result.to_pandas().to_csv("final_evaluation_metrics.csv", index=False)

    THRESHOLDS = {
        "context_precision": 0.70,
        "context_recall": 0.40,
        "faithfulness": 0.40,
        "answer_relevancy": 0.30
    }

    if smoke_test_mode:
        print("\nSmoke test completed successfully. Skipping strict quality gate in CI mode.")
        sys.exit(0)

    print("\n--- Running Quality Gate ---")
    passed = True
    for metric, min_score in THRESHOLDS.items():
        actual_score = result[metric] 
        if actual_score < min_score:
            print(f"FAILED: {metric} scored {actual_score:.4f} (Threshold: {min_score})")
            passed = False
        else:
            print(f"PASSED: {metric} scored {actual_score:.4f} (Threshold: {min_score})")

    if not passed:
        print("\nQuality Gate Failed! Halting pipeline.")
        sys.exit(1) 
    else:
        print("\nQuality Gate Passed! Safe to merge!")
        sys.exit(0) 

if __name__ == "__main__":
    run_evaluations()