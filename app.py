import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure 'RAG workflow' directory is in Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "RAG workflow"))

from retriever import load_vector_store, get_relevant_chunks
from generator import create_prompt
from google import genai

load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic AI Knowledge Assistant")
st.markdown("Query the **Agentic AI Ebook** using a Retrieval-Augmented Generation (RAG) pipeline.")

# Sidebar Controls
st.sidebar.header("⚙️ RAG Configuration")
top_k = st.sidebar.slider("Retrieved Chunks (Top-K):", min_value=1, max_value=8, value=3)

# Load FAISS Index
@st.cache_resource
def init_vectorstore():
    return load_vector_store("faiss_index")

try:
    vectorstore = init_vectorstore()
    st.sidebar.success("FAISS Knowledge Base Loaded")
except Exception as err:
    st.sidebar.error(f"Failed to load FAISS index: {err}")
    vectorstore = None

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chunks" in message and message["chunks"]:
            with st.expander("🔍 View Retrieved Context"):
                for idx, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.text(chunk.page_content)

# Chat Input Box
user_query = st.chat_input("Type your question here (e.g. What is Agentic AI?)...")

if user_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if vectorstore:
        with st.spinner("Searching document & generating answer..."):
            chunks = get_relevant_chunks(vectorstore, user_query, top_k=top_k)
            prompt = create_prompt(user_query, chunks)

        with st.chat_message("assistant"):
            with st.expander("🔍 View Retrieved Context"):
                for idx, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.text(chunk.page_content)

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                    answer = response.text
                except Exception as ex:
                    answer = f"Error generating answer: {ex}"
            else:
                answer = f"**No GEMINI_API_KEY found. Generated Prompt:**\n\n```text\n{prompt}\n```"

            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": chunks
            })
