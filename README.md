# Basic Document Q&A Bot Using RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) based document question-answering system.

The system reads information from PDF documents, splits the extracted text into smaller chunks, creates vector embeddings, stores them in a persistent ChromaDB vector database, retrieves the most relevant chunks for a user question, and generates an answer using a local language model.

The system is designed to answer questions using the provided documents rather than relying on outside knowledge.

---

## Project Overview

This project implements a complete RAG pipeline:

PDF Documents
↓
Text Extraction
↓
Text Cleaning
↓
Chunking
↓
Batched Embeddings
↓
ChromaDB Vector Database
↓
Similarity Search
↓
Retrieved Context
↓
Local LLM
↓
Answer + Source Citation

---

## Features

- PDF document ingestion
- Text extraction using PyMuPDF
- Basic text cleaning
- Word-based chunking with overlap
- Source and page metadata preservation
- Batched embedding generation
- Persistent ChromaDB vector database
- Cosine similarity search
- Configurable top-k retrieval
- Local LLM for answer generation
- Source filename and page citations
- Interactive command-line interface
- Handles questions that cannot be answered from the provided documents

---

## Documents

The project uses five PDF documents:

1. `artificial_intelligence.pdf`
2. `machine_learning.pdf`
3. `cybersecurity.pdf`
4. `renewable_energy.pdf`
5. `electric_vehicles.pdf`

The documents are stored in the `data/` directory.

During indexing, the five documents produced:

- 89 pages with extracted text
- 104 text chunks

---

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10.11 |
| PDF Processing | PyMuPDF |
| Text Embeddings | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| Embedding Dimensions | 384 |
| Vector Database | ChromaDB |
| Similarity Metric | Cosine |
| Local LLM | `Qwen/Qwen2.5-0.5B-Instruct` |
| LLM Framework | Hugging Face Transformers |
| Interface | Command Line Interface (CLI) |

---

## Chunking Strategy

The extracted document text is processed page by page.

Each page is divided into word-based chunks using:

- Chunk size: 500 words
- Chunk overlap: 100 words

The overlap helps preserve context between neighboring chunks.

Each chunk stores metadata including:

- Source filename
- Page number
- Chunk ID

Example chunk ID:

`artificial_intelligence.pdf_page_4_chunk_1`

The implementation keeps chunks within individual pages rather than crossing page boundaries. This also makes source citation straightforward.

---

## Embedding Strategy

The project uses:

`all-MiniLM-L6-v2`

from Sentence Transformers.

The model generates 384-dimensional embeddings.

Embeddings are created in batches using:

`batch_size=32`

This avoids processing every document chunk individually and satisfies the batched embedding requirement.

The embeddings are normalized before being stored in the vector database.

---

## Vector Database

The project uses **ChromaDB** as the vector database.

ChromaDB is configured as a persistent database and stored locally in:

`chroma_db/`

Cosine similarity is used for retrieval.

The database currently contains:

`104 chunks`

The indexing script uses `upsert`, which prevents duplicate entries when the indexing process is run again.

The `chroma_db/` directory is ignored by Git because it is a generated local database and can be recreated by running the indexing script.

---

## Retrieval

For every user question:

1. The question is converted into an embedding.
2. ChromaDB performs similarity search.
3. The top 3 most relevant chunks are retrieved.
4. The retrieved chunks are passed to the local LLM as context.
5. The LLM generates an answer using the retrieved context.

The retrieval value can be changed using the `TOP_K` setting in:

`src/retrieve.py`

---

## Local Language Model

The project uses:

`Qwen/Qwen2.5-0.5B-Instruct`

The model runs locally through Hugging Face Transformers.

No OpenAI API key is required to run the final version of this project.

This makes the project easier to run without depending on a paid external LLM API.

---

## Source Citations

The system preserves the source filename and page number during ingestion and retrieval.

Example:

```text
Sources:
- machine_learning.pdf, Page 2
- machine_learning.pdf, Page 3