from pathlib import Path
import re

import chromadb

from embeddings import load_embedding_model, create_embeddings


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "rag_documents"

TOP_K = 3

# Number of candidates retrieved from ChromaDB before reranking.
CANDIDATE_K = 10


def get_query_keywords(query):
    """
    Extract useful keywords from the question.

    Common question words are removed because they do not
    help distinguish relevant document chunks.
    """

    stop_words = {
        "what",
        "are",
        "is",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "how",
        "why",
        "does",
        "do",
        "what",
        "which",
        "can",
        "with",
        "from",
        "about",
    }

    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())

    return [
        word
        for word in words
        if word not in stop_words and len(word) > 2
    ]


def keyword_score(query, document):
    """
    Calculate a simple keyword relevance score.

    The score represents the proportion of query keywords
    that appear in the retrieved document chunk.
    """

    keywords = get_query_keywords(query)

    if not keywords:
        return 0.0

    document_words = set(
        re.findall(r"\b[a-zA-Z]+\b", document.lower())
    )

    matches = sum(
        1 for keyword in keywords
        if keyword in document_words
    )

    return matches / len(keywords)


def retrieve_chunks(query, top_k=TOP_K):
    """
    Retrieve relevant document chunks using:
    1. Semantic similarity from ChromaDB.
    2. A lightweight keyword relevance boost.

    ChromaDB first retrieves candidate chunks.
    The candidates are then reranked using both
    semantic distance and keyword overlap.
    """

    print("Loading embedding model...")
    model = load_embedding_model()

    query_embedding = create_embeddings(
        model,
        [query],
        batch_size=1,
    )[0]

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=CANDIDATE_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        keyword_relevance = keyword_score(
            query,
            document,
        )

        # Lower distance is better.
        # Keyword overlap provides a small additional boost.
        semantic_score = 1 - distance

        combined_score = (
            0.80 * semantic_score
            + 0.20 * keyword_relevance
        )

        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "keyword_score": keyword_relevance,
                "combined_score": combined_score,
            }
        )

    candidates.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    selected = candidates[:top_k]

    return {
        "documents": [
            [item["document"] for item in selected]
        ],
        "metadatas": [
            [item["metadata"] for item in selected]
        ],
        "distances": [
            [item["distance"] for item in selected]
        ],
    }


if __name__ == "__main__":
    question = input("Enter your question: ")

    results = retrieve_chunks(question)

    print("\nRetrieved chunks:")
    print("-" * 60)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"\nResult {i}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Distance: {distance}")
        print(f"Text:\n{document[:500]}")
        print("-" * 60)