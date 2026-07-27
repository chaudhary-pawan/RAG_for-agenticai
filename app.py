import os
import sys

# Prevent Keras 3 / Transformers conflict
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

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
    page_title="Agentic AI Assistant",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for Big Fonts, Modern Design & Aesthetics
st.markdown("""
<style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Hero Title with Gradient */
    .hero-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Hero Subtitle */
    .hero-subtitle {
        font-size: 1.35rem !important;
        color: #94A3B8;
        margin-bottom: 1.8rem;
    }

    /* Big Question Header */
    .query-header {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #F8FAFC;
        margin-top: 1rem;
    }
    
    /* Large Font Answer Card */
    .answer-card {
        background-color: #1E293B;
        border: 2px solid #334155;
        border-radius: 16px;
        padding: 24px 28px;
        font-size: 1.3rem !important;
        line-height: 1.7;
        color: #F8FAFC;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
        margin-top: 10px;
        margin-bottom: 20px;
    }
    
    /* Large Input Box */
    .stTextInput > div > div > input {
        font-size: 1.25rem !important;
        padding: 14px 18px !important;
        border-radius: 12px !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header Section
st.markdown('<div class="hero-title">⚡ Agentic AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Ask any question to search & summarize the Agentic AI document.</div>', unsafe_allow_html=True)

# Load FAISS Vector Index
@st.cache_resource
def init_vectorstore():
    return load_vector_store("faiss_index")

try:
    vectorstore = init_vectorstore()
except Exception as err:
    st.error(f"Error loading Knowledge Base: {err}")
    vectorstore = None

# Preset Suggestion Buttons
st.markdown("#### 💡 Quick Interactive Questions:")
col1, col2, col3 = st.columns(3)

clicked_query = ""
if col1.button("What is Agentic AI?", use_container_width=True):
    clicked_query = "What is Agentic AI?"
if col2.button("What are the key pillars of Agentic AI?", use_container_width=True):
    clicked_query = "What are the key pillars of Agentic AI?"
if col3.button("How do software agents work?", use_container_width=True):
    clicked_query = "How do software agents work?"

# Query Input Form
with st.form("rag_form"):
    user_query = st.text_input(
        "Ask a question:",
        value=clicked_query if clicked_query else "",
        placeholder="Type your question here and press Search...",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button("🔍 Search & Generate Answer", use_container_width=True)

active_query = clicked_query if clicked_query else (user_query if submit_button else "")

# Process Query and Display Output
if active_query:
    if vectorstore:
        with st.spinner("Searching document and generating answer..."):
            chunks = get_relevant_chunks(vectorstore, active_query, top_k=3)
            prompt = create_prompt(active_query, chunks)

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
                answer = f"**Prompt Prepared for LLM:**\n\n```text\n{prompt}\n```"

        st.markdown("---")
        st.markdown(f'<div class="query-header">❓ Question: {active_query}</div>', unsafe_allow_html=True)
        st.markdown("### 💡 Answer:")
        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

        with st.expander("📄 View Retrieved Document Context"):
            for idx, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {idx}:**")
                st.text(chunk.page_content)
