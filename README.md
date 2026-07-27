# RAG for Agentic AI

A minimal Retrieval-Augmented Generation (RAG) system built to query document knowledge bases.

## Workflow

1. Ingestion: Extract PDF text to Markdown format using PyMuPDF4LLM.
2. Cleaning and Chunking: Clean extracted text and split it into overlapping chunks using LangChain.
3. Embeddings and Vector Store: Convert text chunks into embeddings and store them in a local FAISS index.
4. Retrieval: Perform similarity search to fetch top matching chunks for a user query.
5. Generation: Pass context and query to Gemini API to generate the final response.

## Tech Stack

- Python 3.13
- PyMuPDF4LLM
- LangChain
- Sentence-Transformers (Model: `all-MiniLM-L6-v2`)
- FAISS
- Google GenAI SDK (Model: `gemini-2.5-flash`)

## How to Run

### 1. Build Vector Index
```bash
python Data/knowledge_base.py
```

### 2. Run Interactive RAG Query Engine
```bash
python Data/generator.py
```

## Sample Questions and Answers

### Question 1
What is Agentic AI?

### Answer 1
Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives, shifting technology from reactive to proactive problem-solving.

### Question 2
What is the anatomy of an Agentic AI system?

### Answer 2
The anatomy of an Agentic AI system consists of five core pillars:
- Perception
- Reasoning
- Planning
- Learning
- Execution
