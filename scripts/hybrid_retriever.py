import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)

        self.documents = None
        self.embeddings = None
        self.bm25 = None

    # -------------------------
    # BUILD INDEX
    # -------------------------

    def build_index(self, documents, batch_size=64):

        self.documents = documents

        print("Tokenizing for BM25...")
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print("Encoding embeddings...")

        self.embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True   # ← barre de progression ici
        )

        print("Index built.")

    # -------------------------
    # SAVE / LOAD
    # -------------------------

    def save(self, path):

        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "bm25": self.bm25
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"Index saved → {path}")

    def load(self, path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.bm25 = data["bm25"]

        print("Index loaded.")

    # -------------------------
    # SEARCH
    # -------------------------

    def search(self, query, k=3, alpha=0.5):

        q_embed = self.model.encode(query, normalize_embeddings=True)

        emb_scores = np.dot(self.embeddings, q_embed)
        bm25_scores = self.bm25.get_scores(query.lower().split())

        emb_scores = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-9)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        scores = alpha * emb_scores + (1 - alpha) * bm25_scores

        idx = np.argsort(scores)[::-1][:k]

        return [self.documents[i] for i in idx]