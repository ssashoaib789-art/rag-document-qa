from pathlib import Path
import re

import pymupdf


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def clean_text(text, page_number=None, repeated_lines=None):
    """
    Clean extracted PDF text.

    Removes:
    - extra whitespace
    - standalone page numbers
    - page-number prefixes when they clearly appear
      at the beginning or end of a page
    - repeated header/footer lines detected across pages
    """

    lines = text.splitlines()
    cleaned_lines = []

    for index, line in enumerate(lines):
        line = " ".join(line.split())

        if not line:
            continue

        # Remove standalone page numbers such as "2", "15", "125".
        if re.fullmatch(r"\d+", line):
            continue

        # Remove repeated header/footer lines.
        if repeated_lines and line in repeated_lines:
            continue

        # Remove a page number at the beginning of a line
        # when it matches the current PDF page number.
        if page_number is not None:
            pattern = rf"^{page_number}\s+"
            if index <= 1 or index >= len(lines) - 2:
                line = re.sub(pattern, "", line)

        # Remove a page number at the end of a line
        # when it matches the current PDF page number.
        if page_number is not None:
            pattern = rf"\s+{page_number}$"
            if index <= 1 or index >= len(lines) - 2:
                line = re.sub(pattern, "", line)

        line = line.strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def find_repeated_headers_footers(raw_pages):
    """
    Find lines that repeatedly occur near the beginning or end
    of multiple PDF pages.

    These are treated as likely headers or footers.
    """

    line_counts = {}

    for page_text in raw_pages:
        lines = [
            " ".join(line.split())
            for line in page_text.splitlines()
            if line.strip()
        ]

        # Only examine the first and last few lines of each page.
        boundary_lines = lines[:3] + lines[-3:]

        for line in boundary_lines:
            if len(line) < 4:
                continue

            line_counts[line] = line_counts.get(line, 0) + 1

    # A line appearing on at least 3 pages is likely
    # to be a repeated header or footer.
    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= 3
    }

    return repeated_lines


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

        raw_pages = []

        for page in pdf:
            raw_pages.append(page.get_text("text"))

        repeated_lines = find_repeated_headers_footers(raw_pages)

        for page_number, raw_text in enumerate(raw_pages, start=1):
            cleaned_text = clean_text(
                raw_text,
                page_number=page_number,
                repeated_lines=repeated_lines,
            )

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

    if documents:
        print(f"Source: {documents[0]['source']}")
        print(f"Page: {documents[0]['page']}")
        print(f"Text preview: {documents[0]['text'][:500]}")