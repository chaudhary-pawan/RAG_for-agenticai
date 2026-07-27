import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load_vector_store(index_path="faiss_index"):
    if not os.path.exists(index_path):
        index_path = os.path.join("..", index_path)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def get_relevant_chunks(vectorstore, query, top_k=3):
    return vectorstore.similarity_search(query, k=top_k)

def main():
    vectorstore = load_vector_store("faiss_index")
    query = "What is Agentic AI and how does it work?"
    
    results = get_relevant_chunks(vectorstore, query, top_k=3)

    print(f"Query: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content)
        print()

if __name__ == "__main__":
    main()
