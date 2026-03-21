from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import os 
from dotenv import load_dotenv 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import  Document 
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import pickle
from langchain_community.retrievers import BM25Retriever


load_dotenv()
data_path = os.getenv("DATA_PATH")
chroma_path = os.getenv("CHROMA_PATH")

if not data_path:
    raise ValueError("DATA_PATH is not found!")
if not chroma_path:
    raise ValueError("CHROMA_PATH is not found!")

def load_PDF_document():
    pdf_document_loader = PyPDFDirectoryLoader(data_path)
    return pdf_document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap = 100,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db= Chroma(
        persist_directory=chroma_path, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks=[]
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")

    print("Configuring the BM25 Keyword")
    all_items = db.get(include=["documents", "metadatas"])

    if all_items["documents"]:
        all_chunks = [Document(page_content=txt, metadata=meta) for txt, meta in zip(all_items['documents'], all_items['metadatas'])]
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 3

        with open("bm25_index.pkl", "wb") as f:
            pickle.dump(bm25_retriever, f)
        print("BM25 Index saved to disk successfully!")
    else:
        print("No documents found in DB to build BM25 index")


def calculate_chunk_ids(chunks):
    last_page_id=None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}: {page}"

        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}: {current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks

if __name__ == "__main__":
    documents = load_PDF_document()
    chunks = split_documents(documents)
    add_to_chroma(chunks)