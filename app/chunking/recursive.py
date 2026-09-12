def chunk_text(
    text:str,
    chunk_size:int=1000,
    chunk_overlap:int=200,
) ->list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk overlap should be lesser than chunk size")

    chunks=[]

    start=0
    text_length=len(text)

    while start < text_length:
        end = start + chunk_size

        chunk=text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start+=chunk_size-chunk_overlap
    return chunks