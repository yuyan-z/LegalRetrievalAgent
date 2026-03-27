from pathlib import Path
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from uretriever.utils import load_csv_corpus


class EmbeddingIndex:
    """Multilingual embedding index for semantic search."""

    def __init__(
        self,
        documents: list[dict] | None = None,
        model_name: str = "intfloat/multilingual-e5-base",
        text_field: str = "text",
        citation_field: str = "citation",
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.text_field = text_field
        self.citation_field = citation_field
        self.batch_size = batch_size
        self.device = device

        self.documents: list[dict] = []
        self.embeddings: np.ndarray | None = None

        self.model = SentenceTransformer(model_name, device=device)

        if documents:
            self.build(documents)

    def _doc_to_input(self, text: str) -> str:
        """Format document text for embedding model."""
        text = text.strip()
        # multilingual-e5 requires passage prefix
        return f"passage: {text}"

    def _query_to_input(self, query: str) -> str:
        """Format query text for embedding model."""
        query = query.strip()
        # multilingual-e5 requires query prefix
        return f"query: {query}"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        inputs = [self._doc_to_input(t) for t in texts]
        return self.model.encode(
            inputs,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        inputs = [self._query_to_input(q) for q in queries]
        return self.model.encode(
            inputs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def build(self, documents: list[dict]) -> None:
        self.documents = documents

        texts = [
            str(doc.get(self.text_field, "")) if doc.get(self.text_field) is not None else ""
            for doc in documents
        ]

        if not texts:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        self.embeddings = self.encode_documents(texts)

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("Index not built. Call build() first.")

        if not query.strip():
            return []

        query_emb = self.encode_queries([query])  # (1, dim)

        scores = (self.embeddings @ query_emb.T).squeeze(1)

        if scores.size == 0:
            return []

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            if return_scores:
                doc["_score"] = float(scores[idx])
            results.append(doc)

        return results

    def save(self, path: Path | str) -> None:
        path = Path(path)
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "model_name": self.model_name,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "batch_size": self.batch_size,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "EmbeddingIndex":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            documents=None,
            model_name=data["model_name"],
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
            batch_size=data.get("batch_size", 64),
            device=device,
        )
        instance.documents = data["documents"]
        instance.embeddings = data["embeddings"]
        return instance


def build_embedding_index(
    name: str,
    csv_path: Path,
    index_path: Path,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    model_name: str = "intfloat/multilingual-e5-base",
    device: str | None = None,
) -> EmbeddingIndex:
    if index_path.exists() and not force_rebuild:
        print(f"Loading cached {name} embedding index from {index_path}")
        index = EmbeddingIndex.load(index_path, device=device)
        print(f"  Loaded {len(index.documents):,} documents")
        return index

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating empty index.")
        return EmbeddingIndex(documents=[], model_name=model_name, device=device)

    print(f"Building {name} embedding index from {csv_path}")
    documents = load_csv_corpus(csv_path, max_rows=max_rows)

    if not documents:
        print(f"Warning: No documents loaded. Creating empty index.")
        return EmbeddingIndex(documents=[], model_name=model_name, device=device)

    print(f"\nBuilding embedding index for {len(documents):,} documents...")
    index = EmbeddingIndex(
        documents=documents,
        model_name=model_name,
        text_field="text",
        citation_field="citation",
        device=device,
    )
    print("Index built successfully!")

    print(f"Saving index to {index_path}...")
    index.save(index_path)
    print("Index cached.")

    return index
