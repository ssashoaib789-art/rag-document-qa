# RAG Document Q&A Bot

A beginner-friendly Retrieval-Augmented Generation (RAG) document question-answering system built in Python. The application reads local PDF documents, cleans and chunks their text, creates embeddings, stores them in a persistent ChromaDB vector database, retrieves relevant passages, and generates grounded answers using a local Ollama LLM.

The system is designed to answer questions using only information retrieved from the provided documents and displays the source filename and page number for the retrieved content.

---

## Features

- Reads multiple local PDF documents
- Extracts and cleans PDF text
- Removes repeated headers, footers, and standalone page numbers where possible
- Splits documents into overlapping text chunks
- Generates embeddings in batches
- Stores embeddings persistently in ChromaDB
- Performs semantic similarity search
- Uses keyword overlap as an additional retrieval signal
- Generates answers using a local Ollama LLM
- Provides source filename and page citations
- Displays retrieved source chunks
- Refuses questions when relevant information cannot be found in the provided documents
- Interactive command-line interface (CLI)

---

## Project Documents

The project currently contains five PDF documents:

```text
data/
├── artificial_intelligence.pdf
├── cybersecurity.pdf
├── electric_vehicles.pdf
├── machine_learning.pdf
└── renewable_energy.pdf