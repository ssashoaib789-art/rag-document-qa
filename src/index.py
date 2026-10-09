from pathlib import Path

import chromadb

from ingest import load_documents
from chunk import create_chunks
from embeddings import load_embedding_model, create_embeddings


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "rag_documents"


def create_vector_database():
    """Create embeddings and store them in persistent ChromaDB."""

    print("Loading documents...")
    documents = load_documents()
    print(f"Pages loaded: {len(documents)}")

    print("\nCreating chunks...")
    chunks = create_chunks(documents)
    print(f"Chunks created: {len(chunks)}")

    print("\nLoading embedding model...")
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    print("\nCreating embeddings...")
    embeddings = create_embeddings(
        model,
        texts,
        batch_size=32,
    )

    print(f"Embeddings created: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Remove the previous collection so stale chunks
    # from an older index cannot remain in the database.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Previous collection removed.")
    except Exception:
        print("No previous collection found.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("\nStoring chunks in ChromaDB...")

    ids = [chunk["chunk_id"] for chunk in chunks]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print("\nIndexing completed successfully.")
    print(f"Total chunks in ChromaDB: {collection.count()}")
    print(f"Database location: {CHROMA_DIR}")


if __name__ == "__main__":
    create_vector_database()