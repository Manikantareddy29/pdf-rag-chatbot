import numpy as np

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

_model = None


def get_embedding_model():

    global _model

    if _model is None:

        print(
            "Loading embedding model...",
            flush=True
        )

        _model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print(
            "Embedding model loaded successfully.",
            flush=True
        )

    return _model


def generate_embeddings(texts):

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )