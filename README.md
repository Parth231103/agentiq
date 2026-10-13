# AgentIQ: Autonomous Multi-Agent AI Research Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-ready **Multi-Agent Research Assistant** built with **LangGraph**, **ChromaDB**, **HuggingFace Sentence Transformers**, and **Streamlit**. 

AgentIQ orchestrates specialized autonomous agents (Document Researcher, Web Searcher, and Quantitative Analyst) coordinated by a central Supervisor Router using structured outputs and cyclic state machines.

---

## Architecture Overview

AgentIQ models agent collaboration as a directed cyclic graph with state persistence and conditional routing:

```
                      User Query / Uploads (Streamlit UI)
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │    Supervisor Router     │  ◄── Pydantic Structured Output
                         └─────────────┬────────────┘
                                       │
                     ┌─────────────────┼─────────────────┐
                     ▼                 ▼                 ▼
          ┌──────────────────┐ ┌───────────────┐ ┌────────────────┐
          │ Researcher Agent │ │  WebSearcher  │ │ Analyst Agent  │
          │  (ChromaDB RAG)  │ │ (DuckDuckGo)  │ │ (AST Safe Math)│
          └──────────┬───────┘ └───────┬───────┘ └────────┬───────┘
                     │                 │                  │
                     └─────────────────┼──────────────────┘
                                       ▼
                         ┌──────────────────────────┐
                         │     Synthesizer Node     │  ◄── Cited Output Generation
                         └─────────────┬────────────┘
                                       │
                                       ▼
                            Streamlit Chat Interface
```

---

## Key Technical Highlights

- **LangGraph StateGraph**: Cyclic graph orchestration featuring `add_messages` state reducers, recursion limits, and conditional edges.
- **Supervisor Pattern with Structured Output**: Router inspects conversation context and outputs validated Pydantic models (`AgentRoute`) to dynamically dispatch tasks or transition to synthesis.
- **Local Dense Embeddings & Vector Store**: Integrates `sentence-transformers/all-MiniLM-L6-v2` locally with persistent **ChromaDB**, ensuring fast similarity search without external embedding API costs.
- **Multi-Modal Document Ingestion**: Ingests PDFs, TXT, Markdown, CSV, and live web pages with recursive hierarchical chunking (`RecursiveCharacterTextSplitter`).
- **AST-Hardened Calculator Tool**: Evaluates mathematical expressions using Python's Abstract Syntax Tree (`ast.parse`) parser with strict node allowlisting to eliminate Remote Code Execution (RCE) vulnerabilities.
- **Zero Data Leakage Reset**: Ephemeral session clearing that purges in-memory state and drops ChromaDB collections on demand.

---

## File Structure

The project follows a clean, single-responsibility modular architecture:

```
agentiq/
├── config.py           # Pydantic Settings & centralized get_llm() factory
├── vector_store.py     # Local embeddings, ChromaDB collection & reset operations
├── document_loader.py  # Ingestion engine for PDFs, text files, and web scraping
├── tools.py            # Hardened tool definitions (Knowledge Base, Web, AST Math)
├── agents.py           # Specialist personas and Structured Supervisor Router
├── graph.py            # LangGraph StateGraph, routing conditional edges & Synthesizer
├── app.py              # Streamlit Web UI with multi-agent step visualization
├── requirements.txt    # Production dependencies
├── .env.example        # Environment variable template
├── LICENSE             # MIT License
└── README.md           # Documentation
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/parth231103/agentiq.git
cd agentiq
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your configuration in `.env`:
```env
# Standard OpenAI:
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Or any OpenAI-compatible local proxy (e.g., Ollama / LM Studio / FreeLLM):
# OPENAI_BASE_URL=http://127.0.0.1:31415/v1
# OPENAI_API_KEY=your_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## Resume & Interview Summary

- **Multi-Agent Orchestration**: Engineered a multi-agent system using LangGraph to dynamically route tasks across RAG, web search, and quantitative reasoning agents.
- **Security-First Tool Execution**: Replaced unsafe `eval()` operations with an AST-based parser enforcing strict node validation for mathematical reasoning.
- **Cost-Optimized RAG**: Built local semantic search using HuggingFace sentence transformers and ChromaDB, eliminating embedding API expenses.

---

## License

This project is licensed under the [MIT License](LICENSE).
