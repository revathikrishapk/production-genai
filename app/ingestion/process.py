import uuid

from app.ingestion.loader import load_pdf
from app.chunking.recursive import chunk_text


def process_pdf(file_path: str) -> list[dict]:

    pages = load_pdf(file_path)

    chunks = []

    for page in pages:

        page_chunks = chunk_text(page["text"])

        for chunk in page_chunks:

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        **page["metadata"],
                    },
                }
            )

    return chunks