import json
import threading

import numpy as np

from app.config import VECTOR_FILE, METADATA_FILE

_lock = threading.RLock()


class VectorStore:

    def __init__(self):

        self.vectors = np.empty(
            (0, 0),
            dtype=np.float32
        )

        self.chunks = []

        self.load()

    def load(self):

        with _lock:

            if VECTOR_FILE.exists():

                with open(VECTOR_FILE, "rb") as file:
                    self.vectors = np.load(file)

            if METADATA_FILE.exists():

                with open(
                    METADATA_FILE,
                    "r",
                    encoding="utf-8"
                ) as file:

                    self.chunks = json.load(file)

            if len(self.vectors) != len(self.chunks):

                raise ValueError(
                    "Vector and metadata files have different sizes."
                )

    def save(self):

        with _lock:

            VECTOR_FILE.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            vector_temp = VECTOR_FILE.with_name(
                VECTOR_FILE.name + ".tmp"
            )

            metadata_temp = METADATA_FILE.with_name(
                METADATA_FILE.name + ".tmp"
            )

            with open(vector_temp, "wb") as file:

                np.save(
                    file,
                    self.vectors
                )

            with open(
                metadata_temp,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    self.chunks,
                    file,
                    ensure_ascii=False
                )

            vector_temp.replace(VECTOR_FILE)

            metadata_temp.replace(METADATA_FILE)

    def has_document(self, source, file_hash):

        with _lock:

            return any(
                item["source"] == source
                and item["file_hash"] == file_hash
                for item in self.chunks
            )

    def add_chunks(self, embeddings, metadata):

        with _lock:

            if len(embeddings) != len(metadata):
                raise ValueError(
                    "Embedding and metadata counts do not match."
                )

            if len(self.vectors) == 0:

                self.vectors = embeddings.copy()

            else:

                if self.vectors.shape[1] != embeddings.shape[1]:

                    raise ValueError(
                        "Embedding dimensions do not match."
                    )

                self.vectors = np.vstack([
                    self.vectors,
                    embeddings
                ])

            self.chunks.extend(metadata)

            self.save()

    def remove_document(self, source):

        with _lock:

            indexes = [
                i
                for i, item in enumerate(self.chunks)
                if item["source"] == source
            ]

            if not indexes:
                return 0

            keep = [
                i
                for i, item in enumerate(self.chunks)
                if item["source"] != source
            ]

            if keep:

                self.vectors = self.vectors[keep]

            else:

                self.vectors = np.empty(
                    (0, 0),
                    dtype=np.float32
                )

            self.chunks = [
                self.chunks[i]
                for i in keep
            ]

            self.save()

            return len(indexes)

    
    def search(self, query_embedding, top_k=5, source=None):

        with _lock:

            if len(self.vectors) == 0:
                return []

            query_vector = np.asarray(
                query_embedding,
                dtype=np.float32
            ).reshape(-1)

            if query_vector.shape[0] != self.vectors.shape[1]:
                raise ValueError(
                    "Query embedding dimensions do not match."
                )

            scores = self.vectors @ query_vector

            if source:

                matching_indexes = [
                    i
                    for i, chunk in enumerate(self.chunks)
                    if chunk["source"].lower() == source.lower()
                ]

            else:

                matching_indexes = list(
                    range(len(self.chunks))
                )

            if not matching_indexes:
                return []

            matching_scores = scores[matching_indexes]

            top_positions = np.argsort(
                matching_scores
            )[::-1][:top_k]

            results = []

            for position in top_positions:

                index = matching_indexes[int(position)]

                result = dict(self.chunks[index])

                result["similarity"] = float(
                    scores[index]
                )

                results.append(result)

            return results

    def list_documents(self):

        with _lock:

            documents = {}

            for chunk in self.chunks:

                source = chunk["source"]

                if source not in documents:

                    documents[source] = {
                        "filename": source,
                        "chunks": 0,
                        "pages": set()
                    }

                documents[source]["chunks"] += 1

                documents[source]["pages"].add(
                    chunk["page_number"]
                )

            output = []

            for document in documents.values():

                output.append({
                    "filename": document["filename"],
                    "chunks": document["chunks"],
                    "indexed_pages": len(document["pages"])
                })

            return output

    def count(self):

        with _lock:
            return len(self.chunks)


vector_store = VectorStore()



