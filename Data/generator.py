import os
import sys
from dotenv import load_dotenv
from google import genai
from retriever import load_vector_store, get_relevant_chunks

# Load API key from .env file
load_dotenv()

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def create_prompt(query, chunks):
    context = "\n\n".join([doc.page_content for doc in chunks])
    return f"""Answer the question based ONLY on the provided context below. Do not invent facts. If the answer cannot be found in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{query}

Answer:"""

def generate_response(query, top_k=3):
    vectorstore = load_vector_store("faiss_index")
    chunks = get_relevant_chunks(vectorstore, query, top_k=top_k)
    prompt = create_prompt(query, chunks)

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {e}\n\nFallback Prompt:\n{prompt}"
    else:
        return f"--- PROMPT PREPARED FOR LLM (No API key found) ---\n\n{prompt}"

def main():
    stop_words = ["exit", "quit", "stop", "end", "ok"]
    
    while True:
        query = input("Write your query (or type 'exit' to stop): ").strip()
        
        # Check if user wants to exit
        if query.lower() in stop_words:
            print("Exiting RAG system. Goodbye!")
            break
        
        if not query:
            continue
            
        print(f"\nQuery: {query}\n")
        result = generate_response(query)
        print("\n--- GENERATED ANSWER ---")
        print(result)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
