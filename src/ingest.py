from pathlib import Path
import pymupdf


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def clean_text(text):
    """Clean extracted PDF text without changing its meaning."""
    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def load_documents(data_dir=DATA_DIR):
    """
    Read all PDF files from the data folder.

    Returns a list of page-level records containing:
    - source filename
    - page number
    - cleaned page text
    """

    documents = []

    pdf_files = sorted(data_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        pdf = pymupdf.open(pdf_path)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            cleaned_text = clean_text(text)

            if cleaned_text:
                documents.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": cleaned_text,
                    }
                )

        pdf.close()

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"PDF files found: {len(list(DATA_DIR.glob('*.pdf')))}")
    print(f"Pages with text extracted: {len(documents)}")

    print("\nFirst extracted page:")
    print(f"Source: {documents[0]['source']}")
    print(f"Page: {documents[0]['page']}")
    print(f"Text preview: {documents[0]['text'][:500]}")