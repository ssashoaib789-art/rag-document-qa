from pathlib import Path

import chromadb

from embeddings import load_embedding_model, create_embeddings


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "rag_documents"

# Number of chunks to retrieve
TOP_K = 3


def retrieve_chunks(query, top_k=TOP_K):
    """
    Retrieve the most relevant document chunks
    from the persistent ChromaDB database.
    """

    print("Loading embedding model...")
    model = load_embedding_model()

    # Convert the user question into an embedding
    query_embedding = create_embeddings(
        model,
        [query],
        batch_size=1,
    )[0]

    # Connect to the existing ChromaDB database
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_collection(name=COLLECTION_NAME)

    # Search for the most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    return results


if __name__ == "__main__":
    question = input("Enter your question: ")

    results = retrieve_chunks(question)

    print("\nRetrieved chunks:")
    print("-" * 60)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):
        print(f"\nResult {i}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Text:\n{document[:500]}")
        print("-" * 60)