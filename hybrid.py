from bm25 import Bm25Retriever
from dense import DenseRetriever
from dataset_builder import DatasetBuilder
import numpy as np

class HybridRetriever:

    def __init__(self, corpus, queries, q_rels, bm25: Bm25Retriever, dense: DenseRetriever) -> None:
        self.corpus = corpus
        self.queries = queries
        self.q_rels = q_rels
        self.bm25 = bm25 
        self.dense = dense

        self.doc_ids = list(corpus.keys())

    def _query(self, query_text: str, k: int = 5):
        """Returns top k results for query."""
        bm25_query = self.bm25.encode_query(query_text)
        bm25_scores = self.bm25.index.get_scores(bm25_query)
        
        dense_query = self.dense.encode_query(query_text)
        dense_scores = self.dense.doc_embeddings @ dense_query
        

        bm25_max = np.max(bm25_scores)
        dense_max = np.max(dense_scores)

        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max

        if dense_max > 0:
            dense_scores = dense_scores / dense_max

        hybrid = 0.5 * bm25_scores + 0.5 * dense_scores

        top_indices = np.argsort(hybrid)[::-1][:k]
        return [self.doc_ids[i] for i in top_indices]

    def evaluate(self, query_id: str):
        """Return recall@5, mrr relecant & retrived for each query."""
        relevant = set(self.q_rels[query_id])
        query_text = self.queries[query_id]

        #Recall@5
        retrieved = self._query(query_text, k=5)
        
        hits = len(set(retrieved) & relevant)
        recall_at_5 = hits / len(relevant) if relevant else 0.0

        #MRR
        mrr = 0.0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                mrr = 1 / rank
                break

        return recall_at_5, mrr, relevant, retrieved
    
if __name__ == "__main__":
    builder = DatasetBuilder()
    dataset = builder.run()

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"] 

    bm25 = Bm25Retriever(corpus, queries, qrels)
    dense = DenseRetriever(corpus, queries, qrels)
    hybrid = HybridRetriever(corpus, queries, qrels, bm25, dense)

    for query_id, query_text in queries.items():
        recall_at_5, mrr, relevant, retrieved = hybrid.evaluate(query_id)
        print(f"Query: {query_text}")
        print(f"Relevant docs: {relevant}")
        print(f"Top-5 retrived: {retrieved}")
        print(f"Num relevant: {len(relevant)}")
        print(f"Recall@5: {recall_at_5}")
        print(f"MRR: {mrr}")