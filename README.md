# PDF RAG Chatbot – Backend

## 1. Project Overview

The PDF RAG Chatbot is a Python-based backend application that allows users to upload PDF documents and ask questions about their content.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded PDF files and generate answers using a locally running Large Language Model (LLM).

It supports both:
- Normal PDFs containing selectable text.
- Scanned PDFs containing image-based text using OCR.

The chatbot provides answers with source PDF filenames, page numbers, and retrieved text chunks.

### Main Features

- Upload and process PDF documents.
- Extract text from normal PDFs.
- Extract text from scanned PDFs using OCR.
- Split extracted text into smaller chunks with overlap.
- Generate embeddings using an open-source embedding model.
- Store embeddings and chunk metadata locally.
- Retrieve relevant chunks using vector similarity search.
- Apply lightweight keyword-based reranking.
- Generate answers using a local LLM through Ollama.
- Return source filenames, page numbers, and retrieved chunks.
- Measure API response latency and retrieval evaluation metrics.
- Provide REST APIs through FastAPI.

---

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.15 | Backend programming |
| FastAPI | REST API development |
| Uvicorn | Runs the FastAPI application |
| PyMuPDF | PDF text extraction and page processing |
| Tesseract OCR | Extracts text from scanned PDF pages |
| Pytesseract | Python interface for Tesseract |
| Pillow | Image processing for OCR |
| Sentence Transformers | Generates text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| NumPy | Local vector storage and similarity search |
| Ollama | Runs the local LLM |
| Qwen2.5 1.5B | Answer generation |
| JSON | Stores chunk metadata |
| python-dotenv | Loads environment configuration |

### Why These Technologies?

**FastAPI:** Used to create backend REST APIs and provides automatic API documentation through Swagger UI.

**PyMuPDF:** Extracts text and processes PDF pages.

**Tesseract OCR:** Extracts text from image-based PDF pages.

**Sentence Transformers:** Converts text into numerical vectors called embeddings.

**NumPy:** Stores vectors and performs cosine similarity search without requiring an external vector database.

**Ollama:** Runs the language model locally without depending on a cloud-based LLM API.

**Qwen2.5 1.5B:** Used to generate answers based on retrieved PDF content.

---

## 3. Project Architecture

```text
rag-chatbot/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   └── chat.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── pdf_service.py
│       ├── ocr_service.py
│       ├── chunking_service.py
│       ├── embedding_service.py
│       ├── vector_service.py
│       └── rag_service.py
│
├── documents/
│   └── uploads/
│
├── models/
│
├── vector_store/
│   ├── vectors.npy
│   └── chunks.json
│
├── evaluation/
│   ├── test_questions.json
│   └── evaluate.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### Folder Explanation

**app/**

Contains the complete FastAPI backend application.

**app/routes/**

Contains the API endpoint definitions.

- `documents.py`: Handles document upload, ingestion, listing, and deletion.
- `chat.py`: Handles user questions and chatbot responses.

**app/services/**

Contains the business logic.

- `pdf_service.py`: Extracts text from PDF pages.
- `ocr_service.py`: Performs OCR on scanned pages.
- `chunking_service.py`: Splits extracted text into chunks.
- `embedding_service.py`: Generates text embeddings.
- `vector_service.py`: Stores, retrieves, and deletes vectors.
- `rag_service.py`: Coordinates document ingestion, retrieval, reranking, and answer generation.

**documents/uploads/**

Stores uploaded PDF documents.

**vector_store/**

Stores the locally generated vector index and chunk metadata.

**evaluation/**

Contains the evaluation questions and the script used to measure retrieval quality and response latency.

**models/**

Reserved for local model-related project assets. The embedding model is loaded through Sentence Transformers.

---

## 4. Application Configuration

The application uses a `.env` file to manage configuration values.

Example:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=qwen2.5:1.5b

TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
OCR_LANGUAGE=eng

CHUNK_SIZE=350
CHUNK_OVERLAP=60
TOP_K=3
```

### Configuration Explanation

**EMBEDDING_MODEL**

Defines the embedding model used to convert text into vectors.

We use `all-MiniLM-L6-v2`, which generates 384-dimensional embeddings.

**OLLAMA_MODEL**

Defines the local language model used to generate answers.

We use `qwen2.5:1.5b`.

**TESSERACT_CMD**

Specifies the installed Tesseract executable path.

**OCR_LANGUAGE**

Defines the OCR language. Here, English is used.

**CHUNK_SIZE**

Defines the maximum number of words in a text chunk.

Current value: 350 words.

**CHUNK_OVERLAP**

Defines how many words are repeated between consecutive chunks.

Current value: 60 words.

**TOP_K**

Defines the default number of final relevant chunks retrieved for answering a question.

Current value: 3.

---

## 5. Complete RAG Workflow

The application follows these steps when processing a PDF and answering a question.

### Step 1: PDF Upload

The user uploads a PDF using the document upload API.

The uploaded file is saved in:

```text
documents/uploads/
```

The backend receives the PDF and sends it to the document ingestion service.

### Step 2: PDF Text Extraction

The `pdf_service.py` file uses PyMuPDF to read PDF pages.

For every page:

1. Open the PDF.
2. Extract native text using PyMuPDF.
3. Check the extracted text length.
4. If the extracted text contains fewer than 30 characters, render the page as an image.
5. Send the rendered image to the OCR service.
6. Save the extracted text with its page number and extraction method.

The extraction method is stored as either:

- `text` – native PDF text extraction.
- `ocr` – OCR extraction.

This allows the application to process normal PDFs and fully scanned PDF pages.

**Current implementation note:** OCR is triggered when a page has fewer than 30 extracted characters. Mixed pages containing some native text and additional image-based text may require improved OCR detection.

### Step 3: OCR Processing

The `ocr_service.py` file uses Pytesseract and Tesseract OCR.

The process is:

1. Receive the rendered PDF page image.
2. Pass the image to Tesseract.
3. Extract readable text.
4. Return the extracted text to the PDF service.

This allows the application to process scanned documents where the text is not directly selectable.

### Step 4: Text Chunking

The `chunking_service.py` file splits the extracted page text into smaller chunks.

Current configuration:

- Chunk size: 350 words.
- Overlap: 60 words.

Example:

```text
Chunk 1: Words 1–350
Chunk 2: Words 291–640
Chunk 3: Words 581–930
```

The overlapping words help preserve context between consecutive chunks.

Each chunk stores:

- Extracted text.
- PDF page number.
- Extraction method.

The document filename and file hash are added during ingestion.

### Step 5: Embedding Generation

The `embedding_service.py` file uses Sentence Transformers.

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts every text chunk into a numerical vector.

Each embedding contains 384 dimensions.

The embeddings are normalized before storage, allowing cosine similarity to be calculated efficiently using a vector dot product.

The embedding model is loaded once and reused during the application's runtime.

### Step 6: Vector Storage

The `vector_service.py` file manages local vector storage using NumPy and JSON.

Two files are used:

**vectors.npy**

Stores the numerical embeddings.

**chunks.json**

Stores the associated metadata, including:

- Source filename.
- PDF page number.
- Extracted text.
- Extraction method.
- File hash.

This avoids requiring an external vector database for the current implementation.

The vector store supports:

- Adding document chunks.
- Searching for relevant chunks.
- Listing indexed documents.
- Removing documents.
- Counting indexed chunks.

### Step 7: User Question

The user sends a question through the chat API.

Example:

```json
{
  "question": "What information is mentioned on page 425 of sample-500-pages.pdf?",
  "top_k": 3
}
```

The backend sends the question to the embedding service.

The question is converted into a vector using the same embedding model used for document chunks.

### Step 8: Vector Similarity Search

The application compares the question embedding with the stored document embeddings.

It calculates similarity scores using normalized vector dot products.

The most relevant candidate chunks are retrieved.

For the current implementation, the application retrieves at least 10 candidates when available, before selecting the final top K chunks.

If the question contains a PDF filename, the retrieval service can filter results by that filename.

### Step 9: Reranking

After the initial vector search, the application performs lightweight keyword-based reranking.

It compares:

- Vector similarity score.
- Keyword overlap between the question and retrieved text.

The combined score is calculated as:

```text
Rerank Score =
(0.7 × Similarity Score) +
(0.3 × Keyword Score)
```

The retrieved chunks are sorted by this combined score.

The final top K chunks are selected for answer generation.

This is a lightweight lexical reranking approach, not a cross-encoder reranker.

### Step 10: Context Preparation

The selected chunks are combined into a context.

Each context section includes:

- Source PDF filename.
- Page number.
- Retrieved text.

The context is passed to the language model along with the user's question.

### Step 11: LLM Answer Generation

The application uses Ollama to communicate with the local Qwen model.

The prompt instructs the model to:

1. Answer using the provided context.
2. Avoid inventing information.
3. Avoid guessing incomplete words and acronyms.
4. Clearly state when information is unavailable.
5. Use the actual PDF filenames and page numbers.

The model generates the final answer based on the retrieved context.

### Step 12: Citations and API Response

The application prepares the final response containing:

- Generated answer.
- Retrieved source filename and page references.
- Retrieved chunks.
- Similarity scores.
- Reranking scores.
- Processing time details.

The response is returned as JSON through the chat API.

**Citation limitation:** The current `citations` list represents the selected retrieved source pages. It does not independently verify that every page was explicitly cited in the generated answer.

---

## 6. API Documentation

The application provides REST APIs using FastAPI.

Start the application and open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI provides an interface to test all available endpoints.

### 6.1 Home API

**Method:** GET

**Endpoint:**

```text
/
```

**Purpose:**

Checks whether the backend application is running.

Example response:

```json
{
  "message": "PDF RAG Chatbot Backend is running",
  "status": "success"
}
```

### 6.2 Health Check API

**Method:** GET

**Endpoint:**

```text
/health
```

**Purpose:**

Checks the application health and reports the number of indexed chunks.

Example response:

```json
{
  "status": "healthy",
  "application": "PDF RAG Chatbot",
  "indexed_chunks": 529
}
```

### 6.3 Upload PDF API

**Method:** POST

**Endpoint:**

```text
/api/documents/upload
```

**Purpose:**

Uploads and indexes an individual PDF document.

Processing steps:

1. Receive the uploaded PDF.
2. Save the PDF in the upload directory.
3. Extract text from the PDF.
4. Apply OCR when the page meets the current OCR trigger condition.
5. Split extracted text into chunks.
6. Generate embeddings.
7. Store vectors and metadata.

The endpoint returns the upload and indexing result.

### 6.4 Ingest Folder API

**Method:** POST

**Endpoint:**

```text
/api/documents/ingest-folder
```

**Purpose:**

Processes all PDF files available in the upload directory.

Processing steps:

1. Read PDF files from `documents/uploads/`.
2. Process each PDF.
3. Extract text and perform OCR when required.
4. Generate chunks and embeddings.
5. Store the results in the local vector store.

This is useful when multiple PDFs need to be indexed together.

### 6.5 List Documents API

**Method:** GET

**Endpoint:**

```text
/api/documents
```

**Purpose:**

Returns the list of indexed documents and their chunk counts.

Example response:

```json
{
  "documents": [
    {
      "filename": "sample-500-pages.pdf",
      "chunks": 500,
      "indexed_pages": 500
    }
  ],
  "total_chunks": 500
}
```

The actual response depends on the uploaded documents.

### 6.6 Delete Document API

**Method:** DELETE

**Endpoint:**

```text
/api/documents/{filename}
```

**Purpose:**

Removes a document from the system.

Processing steps:

1. Receive the document filename.
2. Remove its indexed chunks and vectors.
3. Remove the uploaded PDF file.

This allows users to remove documents that are no longer required.

### 6.7 Chat API

**Method:** POST

**Endpoint:**

```text
/api/chat
```

**Purpose:**

Accepts a question and returns an answer based on the indexed PDF documents.

Example request:

```json
{
  "question": "What information is mentioned on page 425 of sample-500-pages.pdf?",
  "top_k": 3
}
```

Example response structure:

```json
{
  "answer": "The document is a machine-generated sample file.",
  "citations": [
    {
      "filename": "sample-500-pages.pdf",
      "page_number": 425
    }
  ],
  "retrieved_chunks": [
    {
      "filename": "sample-500-pages.pdf",
      "page_number": 425,
      "similarity": 0.4823,
      "rerank_score": 0.5626,
      "text": "Extracted PDF content..."
    }
  ],
  "timings": {
    "embedding_seconds": 0.02,
    "vector_search_seconds": 0.00,
    "reranking_seconds": 0.00,
    "context_preparation_seconds": 0.00,
    "llm_generation_seconds": 3.35,
    "formatting_seconds": 0.00,
    "total_seconds": 3.37
  }
}
```

The values above illustrate the response structure; actual values vary by request.

### Chat API Processing

The chat API performs:

1. Question embedding.
2. Optional PDF filename filtering.
3. Vector similarity search.
4. Keyword-based reranking.
5. Context preparation.
6. Ollama LLM generation.
7. Citation and retrieved chunk formatting.
8. Timing calculation.

The `top_k` parameter controls the number of final retrieved chunks.

---

## 7. Evaluation and Performance Testing

An evaluation folder is included to measure retrieval performance and response latency.

Folder structure:

```text
evaluation/
    test_questions.json
    evaluate.py
```

### Evaluation Metrics

**1. Average Latency**

Measures the average time taken by the chat API to return a response.

**2. P95 Latency**

Measures the response-time threshold within which 95% of requests complete, using the evaluation sample.

**3. Recall@K**

Measures whether the expected relevant page appears in the top K retrieved results.

**4. Mean Reciprocal Rank (MRR)**

Measures how highly the expected relevant page is ranked.

**5. Citation Hit Rate**

Checks whether the expected filename and page number appear in the API's returned citations.

**6. Hallucination Evaluation**

Answers are manually reviewed against the retrieved PDF text to identify unsupported information.

The evaluation script automates the numerical metrics. It does not automatically establish whether every generated statement is factually correct.

### Initial Evaluation Results

The first evaluation used one test question and 10 API requests.

| Metric | Result |
|---|---:|
| Successful requests | 10/10 |
| Average latency | 3.74 seconds |
| P95 latency | 5.73 seconds |
| Recall@3 | 100% |
| MRR | 1.00 |
| Expected-page citation hit rate | 100% |

These are initial results from a single question against a synthetic PDF with repeated content. More diverse documents and questions are needed for a reliable overall evaluation.

### Run Evaluation

Make sure the FastAPI application is running.

Open another terminal and execute:

```bash
python evaluation/evaluate.py
```

The script sends requests to the local chat API and prints the evaluation metrics.

---

## 8. Installation and Setup

### Prerequisites

Install or configure:

- Python 3.15.
- Tesseract OCR.
- Ollama.
- The Qwen2.5 1.5B model.

### Step 1: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate the Environment

Windows Command Prompt:

```cmd
venv\Scripts\activate
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Configure Tesseract OCR

Install Tesseract OCR and update its executable path in `.env`.

Example:

```env
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

### Step 5: Configure Ollama

Start Ollama and download the model if it is not already installed:

```bash
ollama pull qwen2.5:1.5b
```

Verify the installed model:

```bash
ollama list
```

### Step 6: Start the FastAPI Application

```bash
python -m uvicorn app.main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

### Step 7: Open Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

Use Swagger UI to upload PDFs, ingest documents, ask questions, and test the APIs.

---

## 9. Testing Scenarios

The following scenarios have been used during initial testing.

### Scenario 1: Normal PDF

Upload a PDF containing selectable text.

Expected result:
- Text extraction succeeds.
- Text is divided into chunks.
- Embeddings are generated.
- Chunks are indexed.

### Scenario 2: Scanned PDF

Upload a scanned PDF containing image-based text.

Expected result:
- The PDF page is rendered as an image when the OCR condition is met.
- Tesseract extracts the text.
- Extracted text is indexed.
- The chatbot can retrieve the extracted text.

### Scenario 3: Specific Page Question

Ask a question about a particular page.

Expected result:
- Relevant content is retrieved.
- The answer refers to the relevant content.
- The response provides the source PDF and page number.

### Scenario 4: Document Summary

Ask the chatbot to summarize a PDF.

Expected result:
- Relevant chunks are retrieved.
- The LLM generates a summary using the available context.

### Scenario 5: Information Not Available

Ask a question about information that does not exist in the PDF.

Expected result:
- The chatbot should clearly state that the requested information is not available.
- It should not invent names, dates, or other facts.

---

## 10. Current Limitations and Future Improvements

The current implementation is a working backend prototype. The following improvements are identified:

1. Improve OCR detection for mixed pages containing both native text and images.
2. Improve hallucination control, especially for incomplete words and technical abbreviations.
3. Validate citations against the actual statements in generated answers.
4. Expand evaluation using multiple meaningful documents and questions.
5. Improve p95 latency and test performance under concurrent requests.
6. Evaluate a dedicated vector database when compatible with the deployment environment.
7. Add a stronger reranking model if performance and compatibility allow.
8. Expand document ingestion testing to the full required dataset.

### Dataset Requirement

The challenge requires at least 10 PDF files, each containing 200 or more pages.

This dataset requirement must be completed and verified before claiming full compliance.

---

## 11. Conclusion

The PDF RAG Chatbot provides a backend solution for extracting information from normal and scanned PDF documents and answering questions using Retrieval-Augmented Generation.

It combines PDF extraction, OCR, text chunking, embedding generation, local vector similarity search, lightweight reranking, and local LLM-based answer generation.

FastAPI provides the REST API layer, while Swagger UI makes it possible to test the application without a separate frontend.

The initial evaluation demonstrates successful API requests, retrieval of the expected page, and an average response latency of 3.74 seconds in the tested environment.

The project can be further improved through larger evaluation datasets, stronger hallucination checks, citation validation, and performance optimization.