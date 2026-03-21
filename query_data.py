import argparse
import time
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM 
from langchain_google_genai import ChatGoogleGenerativeAI, ChatGoogleGenerativeAIError
import json
from get_embedding_function import get_embedding_function
import os
import pickle 
from langchain_classic.retrievers import EnsembleRetriever 
from get_reranker import get_reranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from citation_validator import validate_citations

load_dotenv()
chroma_path = os.getenv("CHROMA_PATH")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def _invoke_with_retries(model, prompt: str, max_retries: int = 3, base_wait_seconds: float = 5.0):
    """Call an LLM with bounded retries for transient quota/rate-limit failures."""
    for attempt in range(1, max_retries + 1):
        try:
            return model.invoke(prompt)
        except Exception as exc:
            error_text = str(exc)
            is_retryable = any(token in error_text for token in ["429", "RESOURCE_EXHAUSTED", "rate limit"])

            if not is_retryable or attempt == max_retries:
                raise

            sleep_for = base_wait_seconds * attempt
            print(
                f"LLM temporarily unavailable (attempt {attempt}/{max_retries}). "
                f"Retrying in {sleep_for:.1f}s..."
            )
            time.sleep(sleep_for)


def load_prompt():
    with open("prompts.json","r") as f:
        prompts = json.load(f)
    return prompts['rag_system_prompt']["template"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

def query_rag(query_text: str):
    if not chroma_path:
        raise ValueError("CHROMA_PATH is not set. Add it to .env or export it in your shell.")

    print("Starting query...")

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    try:
        with open("bm25_index.pkl", "rb") as f:
            bm25_retriever = pickle.load(f)
    except FileNotFoundError:
        print("BM25 index not found. Please run populate_db.py first.")
        return "BM25 index missing."

    vector_retriever = db.as_retriever(search_kwargs={"k": 10})
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    compressor = get_reranker(top_n=5)

    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    results = final_retriever.invoke(query_text)

    if len(results) == 0:
        print("Unable to find any matching context.")
        return "Unable to find matching context."

    context_parts = []
    retrieved_docs = {}
    sources = []
    for doc in results:
        source_id = doc.metadata.get("id", "Unknown")
        context_parts.append(f"[Source ID: {source_id}]\nContent: {doc.page_content}")
        sources.append(source_id)
        retrieved_docs[source_id] = doc.page_content

    context_text = "\n\n---\n\n".join(context_parts)
    allowed_sources_string = "\n".join([f"- {key}" for key in retrieved_docs.keys()])

    prompt_string = load_prompt()
    prompt_template = ChatPromptTemplate.from_template(prompt_string)
    prompt = prompt_template.format(context=context_text, question=query_text, allowed_ids=allowed_sources_string)

    print("\n Generating Response:\n")

    if os.getenv("GITHUB_ACTIONS") == "true":
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        model = OllamaLLM(model="llama3.2")
    
    try:
        raw_response = _invoke_with_retries(model, prompt)
    except ChatGoogleGenerativeAIError as exc:
        if os.getenv("GITHUB_ACTIONS") == "true":
            print(f"Gemini request failed in CI: {exc}")
            if os.getenv("ENABLE_OLLAMA_FALLBACK", "false").lower() == "true":
                print("Falling back to Ollama because ENABLE_OLLAMA_FALLBACK=true")
                fallback_model = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.2"))
                raw_response = _invoke_with_retries(fallback_model, prompt)
            else:
                return (
                    "LLM generation unavailable: Gemini quota/rate limit exceeded. "
                    "Set a valid GEMINI_API_KEY with quota or enable ENABLE_OLLAMA_FALLBACK=true."
                )
        else:
            raise
    except Exception as exc:
        if os.getenv("GITHUB_ACTIONS") == "true" and any(
            token in str(exc) for token in ["429", "RESOURCE_EXHAUSTED", "rate limit"]
        ):
            return (
                "LLM generation unavailable: transient provider quota/rate-limit error. "
                "Retry later or provide a key/model with available quota."
            )
        raise
    

    response_text = raw_response.content if hasattr(raw_response, 'content') else raw_response

    is_safe, final_output = validate_citations(response_text, retrieved_docs)

    if not is_safe:
        print("\n" + "="*50)
        print("BLOCKED UNVERIFIED RESPONSE")
        print("="*50)
        print(f"Raw Output Attempted:\n{response_text}")
        print("\nReason:", final_output)
        print("="*50)
        return final_output

    print(f"Response: {response_text}")
    print(f"\nSources checked: {list(retrieved_docs.keys())}")
    return response_text

if __name__=="__main__":
    main()