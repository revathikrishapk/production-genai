from app.ingestion.loader import load_pdf


documents = load_pdf("data/raw/fds_assessment__7__26MAG0065.pdf")

print(f"Number of pages: {len(documents)}")

for document in documents[:2]:
    print("\n--- PAGE ---")
    print(document["metadata"])
    print(document["text"][:1000])