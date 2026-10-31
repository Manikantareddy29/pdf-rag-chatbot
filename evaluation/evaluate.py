
import json
import math
import time
import urllib.request
import urllib.error
from pathlib import Path
from statistics import mean

API_URL = "http://127.0.0.1:8000/api/chat"
TOP_K = 3
REPEATS = 10

BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "test_questions.json"


def call_chat_api(question):
    payload = json.dumps({
        "question": question,
        "top_k": TOP_K
    }).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed = time.perf_counter() - start

    return result, elapsed


def calculate_recall(chunks, expected_filename, expected_pages):
    retrieved = {
        (chunk["filename"], chunk["page_number"])
        for chunk in chunks[:TOP_K]
    }

    expected = {
        (expected_filename, page)
        for page in expected_pages
    }

    if not expected:
        return 0.0

    return len(retrieved.intersection(expected)) / len(expected)


def calculate_mrr(chunks, expected_filename, expected_pages):
    expected = {
        (expected_filename, page)
        for page in expected_pages
    }

    for rank, chunk in enumerate(chunks[:TOP_K], start=1):
        item = (chunk["filename"], chunk["page_number"])

        if item in expected:
            return 1 / rank

    return 0.0


def calculate_p95(values):
    sorted_values = sorted(values)
    index = math.ceil(0.95 * len(sorted_values)) - 1
    return sorted_values[index]


def main():
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    all_latencies = []
    all_recalls = []
    all_mrrs = []
    citation_hits = 0
    total_tests = 0

    print("\nSTARTING RAG EVALUATION")
    print("=" * 45)

    for test in test_cases:
        question = test["question"]
        filename = test["expected_filename"]
        expected_pages = test["expected_pages"]

        print(f"\nQuestion: {question}")

        for run in range(REPEATS):
            try:
                result, latency = call_chat_api(question)

                chunks = result.get("retrieved_chunks", [])
                citations = result.get("citations", [])

                recall = calculate_recall(
                    chunks, filename, expected_pages
                )

                mrr = calculate_mrr(
                    chunks, filename, expected_pages
                )

                citation_found = any(
                    item.get("filename") == filename
                    and item.get("page_number") in expected_pages
                    for item in citations
                )

                all_latencies.append(latency)
                all_recalls.append(recall)
                all_mrrs.append(mrr)

                if citation_found:
                    citation_hits += 1

                total_tests += 1

                print(
                    f"Run {run + 1}: "
                    f"Latency={latency:.2f}s, "
                    f"Recall@{TOP_K}={recall:.2f}, "
                    f"MRR={mrr:.2f}"
                )

            except urllib.error.URLError as error:
                print(f"API connection failed: {error}")
                print("Make sure FastAPI is running.")
                return

            except Exception as error:
                print(f"Evaluation error: {error}")
                return

    if not all_latencies:
        print("No successful tests were completed.")
        return

    print("\n" + "=" * 45)
    print("FINAL EVALUATION RESULTS")
    print("=" * 45)

    print(f"Total successful requests: {total_tests}")
    print(f"Average latency: {mean(all_latencies):.2f} seconds")
    print(f"P95 latency: {calculate_p95(all_latencies):.2f} seconds")
    print(f"Recall@{TOP_K}: {mean(all_recalls):.2%}")
    print(f"MRR: {mean(all_mrrs):.2f}")

    print(
        "Expected-page citation hit rate: "
        f"{(citation_hits / total_tests) * 100:.2f}%"
    )

    print("\nHallucination check:")
    print("Review the generated answers manually against the PDF.")
    print("This script does not automatically verify factual correctness.")

    print("=" * 45)


if __name__ == "__main__":
    main()
