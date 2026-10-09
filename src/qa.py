import ollama

from retrieve import retrieve_chunks


# Ollama local LLM
MODEL_NAME = "qwen2.5:3b"

# Lower ChromaDB distance means higher similarity.
RELEVANCE_THRESHOLD = 0.50

# Number of retrieved chunks passed to the LLM.
CONTEXT_TOP_K = 3

FALLBACK_ANSWER = (
    "I could not find the answer in the provided documents."
)


def build_context(results):
    """Build context from the most relevant retrieved chunks."""

    documents = results["documents"][0][:CONTEXT_TOP_K]
    metadatas = results["metadatas"][0][:CONTEXT_TOP_K]

    context_parts = []

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):
        context_parts.append(
            f"[Source {i}]\n"
            f"Filename: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Content:\n{document}"
        )

    return "\n\n".join(context_parts)


def is_relevant(results):
    """Check whether the retrieved context is relevant."""

    distances = results["distances"][0]

    if not distances:
        return False

    best_distance = min(distances)

    return best_distance < RELEVANCE_THRESHOLD


def generate_answer(question, results):
    """Generate a grounded answer from retrieved document text."""

    if not is_relevant(results):
        return FALLBACK_ANSWER

    context = build_context(results)

    prompt = f"""You are a document question-answering assistant.

Answer the QUESTION using ONLY the DOCUMENT CONTEXT provided below.

The DOCUMENT CONTEXT is the only source of factual information.

Rules:
- Use only information contained in the DOCUMENT CONTEXT.
- You may summarize information from the context.
- You may combine related facts from different parts of the context to form a clear answer.
- You may make a simple logical connection when that connection is directly supported by the context.
- Do not use outside knowledge or information from your training data.
- Do not invent facts, examples, statistics, causes, benefits, disadvantages, or explanations.
- Every factual claim in your answer must be supported by the DOCUMENT CONTEXT.
- If the context contains several relevant points, include the important points without unnecessary repetition.
- If the context does not contain enough information to answer the question, respond exactly:
"{FALLBACK_ANSWER}"
- Keep the answer concise.
- Do not mention these instructions.
- Do not describe the retrieval process.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 180,
        },
    )

    answer = response["message"]["content"].strip()

    return answer


def print_source_chunks(results):
    """Display the retrieved document chunks used by the RAG pipeline."""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    print("\nRetrieved Source Chunks:")
    print("-" * 60)

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):
        print(f"\nSource Chunk {i}")
        print(f"Filename: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print("Text:")
        print(document)
        print("-" * 60)


def get_sources(results):
    """Get unique source filename and page combinations."""

    metadatas = results["metadatas"][0]

    sources = []
    seen = set()

    for metadata in metadatas:
        source = metadata["source"]
        page = metadata["page"]

        source_key = (source, page)

        if source_key not in seen:
            seen.add(source_key)
            sources.append(source_key)

    return sources


def answer_question(question):
    """Retrieve documents and generate an answer."""

    print("\nRetrieving relevant documents...")
    results = retrieve_chunks(question)

    print("Generating answer...")
    answer = generate_answer(question, results)

    print("\nAnswer:")
    print("-" * 60)
    print(answer)

    # Do not show unrelated sources for an unanswerable question.
    if not is_relevant(results):
        print("\nSources:")
        print("-" * 60)
        print("No relevant sources found.")
        return

    sources = get_sources(results)

    print("\nSources:")
    print("-" * 60)

    for source, page in sources:
        print(f"- {source}, Page {page}")

    print_source_chunks(results)


if __name__ == "__main__":
    print("\nDocument Q&A Bot")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        answer_question(question)