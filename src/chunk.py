from ingest import load_documents


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into word-based chunks with overlap.
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_chunks(documents):
    """
    Convert page-level documents into smaller chunks
    while preserving source and page metadata.
    """

    chunks = []

    for document in documents:
        text_chunks = split_text(document["text"])

        for chunk_number, chunk_text in enumerate(text_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": (
                        f"{document['source']}"
                        f"_page_{document['page']}"
                        f"_chunk_{chunk_number}"
                    ),
                    "source": document["source"],
                    "page": document["page"],
                    "text": chunk_text,
                }
            )

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = create_chunks(documents)

    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst chunk:")
        print(f"Chunk ID: {chunks[0]['chunk_id']}")
        print(f"Source: {chunks[0]['source']}")
        print(f"Page: {chunks[0]['page']}")
        print(f"Words: {len(chunks[0]['text'].split())}")
        print(f"\nText preview:\n{chunks[0]['text'][:500]}")