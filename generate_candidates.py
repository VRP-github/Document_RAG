import pandas as pd
import re
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

def main():
    target_questions = 100

    loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("Error: No PDFs found in the 'data' folder!")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    chunks = chunks[:target_questions]

    llm = ChatOllama(model="llama3.2", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert test creator. Based ONLY on the following context, generate 1 highly specific question and its correct answer.
    
    Context: {context}

    Format your response EXACTLY like this:
    Question: [Your question here]
    Answer: [Your answer here]
    """)

    chain = prompt | llm
    data = {"question": [], "contexts": [], "ground_truth": []}

    print(f"Generating up to {target_questions} Q&A pairs")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}...")
        try:
            response = chain.invoke({"context": chunk.page_content}).content

            question_match = re.search(r"Question:\s*(.*)", response, re.IGNORECASE)
            answer_match = re.search(r"Answer:\s*(.*)", response, re.IGNORECASE)
            
            if question_match and answer_match:
                q = question_match.group(1).strip()
                a = answer_match.group(1).strip()
                
                data["question"].append(q)
                data["contexts"].append([chunk.page_content]) 
                data["ground_truth"].append(a)
            else:
                print(f"      Skipped chunk {i+1}: LLM failed to follow the Question/Answer format.")
                
        except Exception as e:
            print(f"      Error on chunk {i+1}: {e}")

    df = pd.DataFrame(data)
    df.to_csv("candidate_dataset.csv", index=False)

if __name__ == "__main__":
    main()