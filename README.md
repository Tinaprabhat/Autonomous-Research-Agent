
# Autonomous Research Agent

A **fully local AI research system** that autonomously retrieves, analyzes, and synthesizes academic knowledge using **LLMs, hybrid retrieval, and reranking pipelines**.

Unlike traditional RAG chatbots, this project implements a **complete research workflow**:

* ingest academic papers
* build a semantic knowledge base
* retrieve relevant research
* rerank results for precision
* generate grounded answers with a local LLM

The entire system runs **locally without cloud AI APIs**.

---

# Problem Statement

Large Language Models often produce **hallucinated answers** when responding to technical or research questions.

The goal of this project is to build a **grounded AI research assistant** that:

1. retrieves relevant academic knowledge
2. filters and ranks the most relevant information
3. generates concise answers strictly based on retrieved context

This architecture significantly improves **faithfulness and reliability** compared to naive LLM generation.

---

# System Overview

The system implements a **multi-stage research pipeline** designed to maximize answer quality.

```
User Query
    │
    ▼
Planner Agent
    │
    ▼
Hybrid Retrieval
(Vector Search + BM25)
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Context Construction
    │
    ▼
Local LLM Generation
    │
    ▼
Research Answer
```

This layered approach reduces hallucination risk by ensuring that the LLM only operates on **verified retrieved context**.

---

# Architecture

## High-Level System Design

```
              ┌────────────────────┐
              │   User Query        │
              └─────────┬──────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Planner Agent  │
               └─────────┬───────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   Hybrid Retrieval      │
            │  Vector + BM25 Search   │
            └─────────┬──────────────┘
                      │
                      ▼
           ┌──────────────────────────┐
           │  Cross-Encoder Reranker  │
           └─────────┬────────────────┘
                     │
                     ▼
            ┌───────────────────────┐
            │  Context Construction │
            └─────────┬─────────────┘
                      │
                      ▼
             ┌─────────────────────┐
             │ Local LLM (Mistral) │
             └─────────┬───────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  Final Answer     │
              └───────────────────┘
```

---

# Core Components

## Planner Agent

The planner generates a structured execution plan for a research query.

Example plan:

```
task: research_query
steps:
 - retrieve_documents
 - analyze_documents
 - generate_summary
```

The planner allows the system to evolve into a **multi-agent research workflow**. 

---

# Research Agent

The Research Agent orchestrates the entire reasoning pipeline:

1. retrieve candidate documents
2. rerank the results
3. construct contextual prompt
4. generate grounded answer

This ensures answers are derived strictly from retrieved research content. 

---

# Knowledge Ingestion Pipeline

Academic papers are automatically collected and processed.

Pipeline:

```
arXiv Search
    ↓
PDF Download
    ↓
Text Extraction
    ↓
Semantic Chunking
    ↓
Embedding Generation
    ↓
Vector Index Construction
```

The ingestion system supports building **large research knowledge bases locally**. 

---

# Semantic Chunking

Documents are chunked using a **semantic-aware pipeline**.

Process:

1. paragraph segmentation
2. semantic merging using cosine similarity
3. recursive character splitting

This approach preserves **topical coherence within chunks**, improving retrieval quality. 

---

# Hybrid Retrieval

The retrieval layer combines:

* **vector similarity search**
* **keyword-based BM25 retrieval**

Hybrid retrieval improves recall by capturing both semantic and lexical relevance. 

---

# Cross-Encoder Reranking

Candidate documents are reranked using a **cross-encoder transformer model**.

Cross-encoders evaluate query-document pairs directly, improving precision in top results. 

---

# Vector Database

The system stores document embeddings in a **FAISS vector index**.

Features:

* efficient similarity search
* persistent metadata storage
* scalable document retrieval

Each vector is linked with metadata for contextual reconstruction. 

---

# Local LLM Engine

Responses are generated using a locally hosted LLM.

Features:

* streaming generation
* response caching
* timeout handling
* persistent model loading

The system communicates with the model through Ollama’s local API. 

---

# Model Optimization

A CPU optimization pipeline automatically tunes LLM inference parameters.

The optimizer performs:

* hardware detection
* grid search for optimal parameters
* latency benchmarking
* model configuration generation

This allows efficient operation even on **CPU-only systems**. 

---

# Evaluation Framework

The project includes a built-in evaluation suite measuring four key dimensions:

| Metric       | Description                           |
| ------------ | ------------------------------------- |
| Relevance    | Does the answer address the question? |
| Faithfulness | Does the answer stick to the context? |
| Conciseness  | Is the answer focused and minimal?    |
| Latency      | How fast is the response?             |

Each test case is scored **0–12 points**.

Example evaluation summary:

```
Average Score: 9.5 / 12
Average Latency: 33 seconds
Rating: Good
```

This ensures continuous monitoring of system performance. 

---

# Repository Structure

```
backend/
 ├── agents
 │    ├── planner.py
 │    └── research_agent.py
 │
 ├── ingestion
 │    ├── arxiv_fetcher.py
 │    ├── pdf_parser.py
 │    ├── chunking.py
 │    └── metadata_extractor.py
 │
 ├── rag_pipeline
 │    ├── embedder.py
 │    ├── vector_store.py
 │    ├── retriever.py
 │    ├── hybrid_retriever.py
 │    └── reranker.py
 │
 ├── llm
 │    └── local_llm.py
 │
 └── api.py
```

---

# Example Usage

## Build Vector Database

```
python build_embeddings.py
```

This script generates embeddings and builds the FAISS index. 

---

## Run Research Agent

```
python test_agent.py
```

Example query:

```
What are the main approaches to meta-learning?
```

The agent retrieves relevant papers and produces a concise research summary. 

---

## Run Evaluation

```
python evaluate_llm.py
```

Runs benchmark queries and evaluates system performance.

---

# Technologies

Core technologies used in the system:

* Python
* FAISS
* Sentence Transformers
* Cross-Encoder models
* Ollama
* PyMuPDF
* BM25

---

# Design Philosophy

This project follows three guiding principles:

### Grounded Generation

The LLM must rely strictly on retrieved context.

### Retrieval Quality First

Better retrieval improves downstream reasoning.

### Fully Local AI

The entire research pipeline runs without external APIs.

---

# Potential Extensions

Future improvements could include:

* autonomous planning agents
* multi-hop research pipelines
* citation-aware answers
* trust scoring for sources
* distributed vector databases

---

# Author

Tina Prabhat
B.Tech Computer Science

Areas of interest:

* AI systems engineering
* retrieval augmented generation
* research automation

---

# License

MIT License

---



