from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the sentence-transformer embedding model."""
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, texts, batch_size=32):
    """
    Create embeddings in batches.

    The assignment requires batched embedding generation,
    so chunks are processed in groups instead of one at a time.
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


if __name__ == "__main__":
    model = load_embedding_model()

    test_texts = [
        "Artificial intelligence allows computers to perform tasks that normally require human intelligence.",
        "Machine learning enables systems to learn patterns from data.",
    ]

    embeddings = create_embeddings(model, test_texts)

    print("\nEmbedding test successful.")
    print(f"Number of texts: {len(test_texts)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}") 