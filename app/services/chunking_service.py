from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(pages):

    all_chunks = []

    for page in pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = min(
                start + CHUNK_SIZE,
                len(words)
            )

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text.strip():

                all_chunks.append({
                    "text": chunk_text,
                    "page_number": page["page_number"],
                    "extraction_method": page["extraction_method"]
                })

            if end == len(words):
                break

            start = end - CHUNK_OVERLAP

    return all_chunks