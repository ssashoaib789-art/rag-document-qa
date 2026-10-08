from transformers import pipeline, GenerationConfig

from retrieve import retrieve_chunks


# Local LLM
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


print("Loading local LLM...")

generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
)

generation_config = GenerationConfig(
    max_new_tokens=120,
    do_sample=False,
)

# Prevent BPE tokenizer cleanup warning
generator.tokenizer.clean_up_tokenization_spaces = False

print("Local LLM loaded successfully.")


def build_context(results):
    """Build context from retrieved document chunks."""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

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


def generate_answer(question, results):
    """Generate an answer using only the retrieved context."""

    context = build_context(results)

    prompt = f"""You are a document question-answering assistant.

Use ONLY the information in the document context below.

Rules:
- Answer the question directly.
- Keep the answer short: 1 to 3 sentences.
- Give only the information needed to answer the question.
- Use only information supported by the context.
- Do not use outside knowledge.
- Do not invent facts.
- Do not summarize or describe the source document.
- Stop immediately after answering the question.
- If the answer is not supported by the context, say exactly:
"I could not find the answer in the provided documents."
- Do not include citations. The program will add them automatically.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    result = generator(
        prompt,
        generation_config=generation_config,
        return_full_text=False,
    )

    answer = result[0]["generated_text"].strip()

    # Remove unwanted continuation after the main answer.
    stop_phrases = [
        "\nI could not find the answer",
        "The chapter thus",
        "Readers with a background",
        "Keywords:",
    ]

    for phrase in stop_phrases:
        if phrase in answer:
            answer = answer.split(phrase)[0].strip()

    return answer


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

    sources = get_sources(results)

    print("\nAnswer:")
    print("-" * 60)
    print(answer)

    print("\nSources:")
    print("-" * 60)

    for source, page in sources:
        print(f"- {source}, Page {page}")


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