import hashlib

import ollama

from app.config import OLLAMA_MODEL, TOP_K

from app.services.pdf_service import extract_pdf_pages
from app.services.chunking_service import create_chunks
from app.services.embedding_service import generate_embeddings
from app.services.vector_service import vector_store
import re
from app.config import OLLAMA_MODEL, TOP_K
import time


def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            sha256.update(block)

    return sha256.hexdigest()


def ingest_pdf(pdf_path):

    source = pdf_path.name

    file_hash = calculate_file_hash(pdf_path)

    if vector_store.has_document(source, file_hash):

        return {
            "filename": source,
            "status": "already_indexed",
            "chunks": 0
        }

    print(
        f"Processing PDF: {source}",
        flush=True
    )

    pages = extract_pdf_pages(pdf_path)

    if not pages:

        raise ValueError(
            "No readable text found in this PDF."
        )

    chunks = create_chunks(pages)

    if not chunks:

        raise ValueError(
            "No text chunks were generated."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    metadata = []

    for chunk in chunks:

        metadata.append({
            "source": source,
            "file_hash": file_hash,
            "page_number": chunk["page_number"],
            "extraction_method": chunk["extraction_method"],
            "text": chunk["text"]
        })

    vector_store.remove_document(source)

    vector_store.add_chunks(
        embeddings,
        metadata
    )

    print(
        f"Indexed {len(chunks)} chunks from {source}",
        flush=True
    )

    return {
        "filename": source,
        "status": "indexed",
        "total_pages": len(pages),
        "total_chunks": len(chunks)
    }


def ingest_folder(folder_path):

    results = []

    pdf_files = sorted(
        folder_path.glob("*.pdf")
    )

    for pdf_path in pdf_files:

        try:

            result = ingest_pdf(pdf_path)

            results.append(result)

        except Exception as e:

            results.append({
                "filename": pdf_path.name,
                "status": "failed",
                "error": str(e)
            })

    return {
        "total_pdfs": len(pdf_files),
        "results": results
    }


def answer_question(question, top_k=TOP_K):

    total_start = time.perf_counter()

    start = time.perf_counter()

    query_embedding = generate_embeddings([question])[0]

    embedding_time = time.perf_counter() - start

    print(f"Embedding time: {embedding_time:.2f} seconds", flush=True)

    pdf_match = re.search(
        r'([\w.-]+\.pdf)',
        question,
        re.IGNORECASE
    )

    source = pdf_match.group(1) if pdf_match else None

    start = time.perf_counter()

    candidate_k = max(top_k * 3, 10)

    retrieved_chunks = vector_store.search(
        query_embedding,
        top_k=candidate_k,
        source=source
    )

    vector_search_time = time.perf_counter() - start

    print(
        f"Vector search time: {vector_search_time:.2f} seconds",
        flush=True
    )

    if not retrieved_chunks:

        if source:
            message = (
                f"No indexed content found for {source}. "
                "Please check the filename or upload the PDF."
            )
        else:
            message = (
                "No documents have been indexed yet. "
                "Please upload a PDF first."
            )

        total_time = time.perf_counter() - total_start

        return {
            "answer": message,
            "citations": [],
            "retrieved_chunks": [],
            "timings": {
                "embedding_seconds": round(embedding_time, 2),
                "vector_search_seconds": round(vector_search_time, 2),
                "total_seconds": round(total_time, 2)
            }
        }

    start = time.perf_counter()

    stop_words = {
        "the", "and", "for", "are", "was",
        "what", "when", "where", "which",
        "who", "how", "does", "this", "that",
        "with", "from", "have", "has", "can",
        "you", "your", "about", "into", "main",
        "points", "summarize", "summary"
    }

    query_words = set(
        re.findall(r'\b[a-zA-Z0-9]{3,}\b', question.lower())
    )

    query_words = query_words - stop_words

    for chunk in retrieved_chunks:

        chunk_words = set(
            re.findall(
                r'\b[a-zA-Z0-9]{3,}\b',
                chunk["text"].lower()
            )
        )

        if query_words:
            matched_words = query_words.intersection(chunk_words)

            keyword_score = len(matched_words) / len(query_words)
        else:
            keyword_score = 0.0

        similarity_score = chunk["similarity"]

        chunk["rerank_score"] = (
            0.7 * similarity_score +
            0.3 * keyword_score
        )

    retrieved_chunks.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    retrieved_chunks = retrieved_chunks[:top_k]

    reranking_time = time.perf_counter() - start

    print(
        f"Reranking time: {reranking_time:.2f} seconds",
        flush=True
    )

    start = time.perf_counter()

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Source: {chunk['source']}
Page: {chunk['page_number']}
Content:
{chunk['text'][:600]}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the provided context.

Instructions:
1. Do not invent information.
2. If the answer is not available in the context, say so clearly.
3. Give a simple and clear answer.
4. Add citations using this format:
   [Source: filename.pdf, Page: 10]
5. Use the actual source filenames and page numbers provided.
6. Do not create fake citations.

Context:

{context}

User Question:

{question}

Answer:
"""

    context_time = time.perf_counter() - start

    print(
        f"Context preparation time: {context_time:.2f} seconds",
        flush=True
    )

    start = time.perf_counter()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 150,
            "num_ctx": 2048
        },
        keep_alive="10m"
    )

    llm_time = time.perf_counter() - start

    load_time = getattr(response, "load_duration", 0) or 0
    prompt_time = getattr(response, "prompt_eval_duration", 0) or 0
    prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
    generation_time = getattr(response, "eval_duration", 0) or 0
    generated_tokens = getattr(response, "eval_count", 0) or 0

    prompt_eval_rate = (
        prompt_tokens / (prompt_time / 1e9)
        if prompt_time > 0 else 0
    )

    print("-" * 40, flush=True)
    print("OLLAMA PERFORMANCE DETAILS", flush=True)
    print(f"Ollama total time: {llm_time:.2f}s", flush=True)
    print(f"Model loading time: {load_time / 1e9:.2f}s", flush=True)
    print(f"Prompt processing time: {prompt_time / 1e9:.2f}s", flush=True)
    print(f"Input prompt tokens: {prompt_tokens}", flush=True)
    print(f"Prompt processing speed: {prompt_eval_rate:.2f} tokens/sec", flush=True)
    print(f"Token generation time: {generation_time / 1e9:.2f}s", flush=True)
    print(f"Generated tokens: {generated_tokens}", flush=True)

    if generation_time > 0:
        tokens_per_second = generated_tokens / (generation_time / 1e9)
        print(f"Generation speed: {tokens_per_second:.2f} tokens/sec", flush=True)

    print("-" * 40, flush=True)

    answer = response.message.content

    start = time.perf_counter()

    citations = []
    seen = set()

    for chunk in retrieved_chunks:

        citation = (
            chunk["source"],
            chunk["page_number"]
        )

        if citation not in seen:
            seen.add(citation)

            citations.append({
                "filename": chunk["source"],
                "page_number": chunk["page_number"]
            })

    formatted_chunks = []

    for chunk in retrieved_chunks:

        formatted_chunks.append({
            "filename": chunk["source"],
            "page_number": chunk["page_number"],
            "similarity": round(chunk["similarity"], 4),
            "rerank_score": round(chunk["rerank_score"], 4),
            "text": chunk["text"]
        })

    formatting_time = time.perf_counter() - start

    print(
        f"Response formatting time: {formatting_time:.2f} seconds",
        flush=True
    )

    total_time = time.perf_counter() - total_start

    print("=" * 40, flush=True)
    print("RAG PERFORMANCE SUMMARY", flush=True)
    print(f"Embedding: {embedding_time:.2f}s", flush=True)
    print(f"Vector search: {vector_search_time:.2f}s", flush=True)
    print(f"Reranking: {reranking_time:.2f}s", flush=True)
    print(f"Context preparation: {context_time:.2f}s", flush=True)
    print(f"LLM generation: {llm_time:.2f}s", flush=True)
    print(f"Formatting: {formatting_time:.2f}s", flush=True)
    print(f"TOTAL: {total_time:.2f}s", flush=True)
    print("=" * 40, flush=True)

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": formatted_chunks,
        "timings": {
            "embedding_seconds": round(embedding_time, 2),
            "vector_search_seconds": round(vector_search_time, 2),
            "reranking_seconds": round(reranking_time, 2),
            "context_preparation_seconds": round(context_time, 2),
            "llm_generation_seconds": round(llm_time, 2),
            "formatting_seconds": round(formatting_time, 2),
            "total_seconds": round(total_time, 2)
        }
    }
