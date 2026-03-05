from typing import List
from chromadb import PersistentClient
from chromadb.api import ClientAPI

from pathlib import Path

from ingest.chunker import Chunk

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)  # src/storage/vector_store.py → project root
DEFAULT_CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma")


class VectorStore:
    def __init__(self, client: ClientAPI | None = None, collection_name: str = "notes"):
        self._client = client if client else PersistentClient(path=DEFAULT_CHROMA_PATH)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) == 0 or len(embeddings) == 0:
            return

        ids = [f"{chunk.source}-{index}" for index, chunk in enumerate(chunks)]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "note_title": chunk.note_title,
                "heading": chunk.heading,
                "tags": ",".join(chunk.tags),
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,  # type: ignore
            embeddings=embeddings,  # type: ignore
        )

    def reset(self) -> None:
        self._client.delete_collection(name=self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name, metadata={"hnsw:space": "cosine"}
        )

    def search(self, embedding: List[float], top_k: int):
        if not embedding or len(embedding) == 0:
            return []

        response = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        distances = response.get("distances", [])

        if not documents or not metadatas or not distances:
            return []

        results = []
        for document, metadata, distance in zip(
            documents[0], metadatas[0], distances[0]
        ):
            results.append(
                {
                    "text": document,
                    "source": metadata.get("source", ""),
                    "note_title": metadata.get("note_title", ""),
                    "heading": metadata.get("heading", ""),
                    "tags": metadata.get("tags", ""),
                    "distance": distance,
                }
            )

        return results
