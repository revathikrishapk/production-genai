import json
from pathlib import Path


DOCUMENTS_PATH = Path(
    "data/processed/documents.json"
)


def load_documents():

    with open(
        DOCUMENTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def build_dataset():

    documents = load_documents()

    print()
    print("=" * 80)
    print("DOCUMENT CHUNKS")
    print("=" * 80)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        chunk_id = document["id"]

        metadata = document[
            "metadata"
        ]

        page = metadata.get(
            "page",
            "unknown",
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        text = (
            document["text"]
            .replace("\n", " ")
            .strip()
        )

        print()
        print("-" * 80)

        print(
            f"CHUNK #{index}"
        )

        print(
            f"ID     : {chunk_id}"
        )

        print(
            f"SOURCE : {source}"
        )

        print(
            f"PAGE   : {page}"
        )

        print()

        print(text)

        print()


if __name__ == "__main__":
    build_dataset()