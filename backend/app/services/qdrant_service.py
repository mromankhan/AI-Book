from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from typing import List, Optional
import uuid

from app.config import settings

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

VECTOR_SIZE = 1536  # text-embedding-3-small dimension


def ensure_collection():
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection_name not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_chunks(chunks: List[dict], embeddings: List[List[float]]):
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "content": chunk["content"],
                "chapter": chunk["chapter"],
                "section": chunk["section"],
                "chunk_id": chunk.get("chunk_id", point_id),
            }
        ))

    # Batch upsert in groups of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=batch,
        )


def search_similar(query_vector: List[float], limit: int = 5, chapter_filter: Optional[str] = None):
    search_filter = None
    if chapter_filter:
        search_filter = Filter(
            must=[FieldCondition(key="chapter", match=MatchValue(value=chapter_filter))]
        )

    results = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        limit=limit,
        query_filter=search_filter,
        with_payload=True,
    )

    return [
        {
            "content": point.payload["content"],
            "chapter": point.payload["chapter"],
            "section": point.payload["section"],
            "score": point.score,
        }
        for point in results.points
    ]


def delete_collection():
    client.delete_collection(collection_name=settings.qdrant_collection_name)
