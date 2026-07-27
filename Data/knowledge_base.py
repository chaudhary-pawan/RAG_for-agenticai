import os
import re
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def read_pdf(pdf_path):
    return pymupdf4llm.to_markdown(pdf_path)

def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_vector_store(chunks, save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

def main():
    pdf_path = "Ebook-Agentic-AI.pdf"
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join("..", pdf_path)

    raw_text = read_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    create_vector_store(chunks, save_path="faiss_index")
    print("Knowledge base created successfully!")

if __name__ == "__main__":
    main()
