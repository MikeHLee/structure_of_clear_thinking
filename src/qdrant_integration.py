import numpy as np
from typing import Dict, List, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchParams
import uuid

class QdrantHDCIndex:
    """
    Qdrant-based HNSW index for O(d log V) hypervector similarity search.
    Mitigates the O(V*d) exact search bottleneck for Freebase-scale KGs.
    """
    
    def __init__(self, collection_name: str = "hdc_vectors", dim: int = 4096, in_memory: bool = True):
        self.collection_name = collection_name
        self.dim = dim
        
        # Use in-memory Qdrant for easy testing/prototyping.
        # Can easily swap to persistent or cloud Qdrant via QdrantClient(url="...")
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path="./qdrant_data")
            
        self._setup_collection()
        self._name_to_id: Dict[str, str] = {}
        
    def _setup_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim, 
                    distance=Distance.COSINE
                )
            )
            
    def insert_vectors(self, named_vectors: Dict[str, np.ndarray]):
        """Batch insert hypervectors into Qdrant."""
        points = []
        for name, vector in named_vectors.items():
            point_id = str(uuid.uuid4())
            self._name_to_id[name] = point_id
            
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={"name": name}
            ))
            
        if points:
            # Upsert in batches to avoid payload limits
            batch_size = 1000
            for i in range(0, len(points), batch_size):
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points[i:i+batch_size]
                )
                
    def search(self, query_vector: np.ndarray, top_k: int = 10, ef_search: int = 128) -> List[Tuple[str, float]]:
        """
        Search for most similar vectors using HNSW index.
        Time complexity: O(d * log V)
        """
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            search_params=SearchParams(hnsw_ef=ef_search)
        ).points
        
        return [(hit.payload["name"], hit.score) for hit in search_result]

def demo_qdrant():
    print("Initializing Qdrant In-Memory HNSW Index...")
    index = QdrantHDCIndex(dim=1024)
    
    print("Generating random vectors...")
    vocab_size = 5000
    dim = 1024
    vectors = {f"entity_{i}": np.random.randn(dim) for i in range(vocab_size)}
    
    print(f"Inserting {vocab_size} vectors...")
    index.insert_vectors(vectors)
    
    query = np.random.randn(dim)
    print("Executing approximate nearest neighbor search...")
    import time
    start = time.perf_counter()
    results = index.search(query, top_k=5)
    elapsed = time.perf_counter() - start
    
    print(f"Search completed in {elapsed*1000:.2f}ms")
    for name, score in results:
        print(f"  {name}: {score:.4f}")

if __name__ == "__main__":
    demo_qdrant()
