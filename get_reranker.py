from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def get_reranker(top_n: int = 3):
    
    model = HuggingFaceCrossEncoder(model_name ="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=model,  top_n=top_n)

    return reranker 