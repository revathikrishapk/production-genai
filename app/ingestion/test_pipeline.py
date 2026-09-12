from app.ingestion.process import process_pdf


chunks = process_pdf("data/raw/fds_assessment__7__26MAG0065.pdf")

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:5]):

    print("\n--------------")
    print(f"CHUNK {i + 1}")
    print("----------------")

    print(chunk["metadata"])
    print(chunk["text"][:500])