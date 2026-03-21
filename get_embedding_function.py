from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformer.modeling_utils").setLevel(logging.ERROR)

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings