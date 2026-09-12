import json
from pathlib import Path


DEFAULT_PATH = "data/processed/documents.json"


def save_documents(
    documents: list[dict],
    path: str = DEFAULT_PATH,
):
    """
    Save processed document chunks to JSON.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved {len(documents)} documents to "
        f"{output_path}"
    )


def load_documents(
    path: str = DEFAULT_PATH,
) -> list[dict]:
    """
    Load processed document chunks from JSON.
    """

    input_path = Path(path)

    if not input_path.exists():

        raise FileNotFoundError(
            f"Processed documents not found: {path}\n"
            f"Run the indexing pipeline first."
        )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:

        documents = json.load(file)

    if not isinstance(documents, list):

        raise ValueError(
            "Processed documents file must contain a list."
        )

    return documents