from dataset_builder import DatasetBuilder
from rank_bm25 import BM25Okapi
import numpy as np

class Bm25Retriever:

    def __init__(self, corpus, queries, q_rels) -> None:
        """
        corpus: {doc_id: text}
        queries: {query_id: text}
        qrels: {query_id: [doc_ids]}
        """
        self.corpus = corpus
        self.queries = queries
        self.q_rels = q_rels

        self.doc_ids = list(corpus.keys())
        self.index = self._build_index()

    def _build_index(self):
        documents = [self.corpus[doc_id] for doc_id in self.doc_ids]
        tokenised_docs = [self.tokenise(doc) for doc in documents]
        return BM25Okapi(tokenised_docs)

    def _query(self, query_text: str, k: int = 5):
        """Returns top k results for query."""
        query_tokens = self.tokenise(query_text)
        scores = self.index.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:k]

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

    @staticmethod
    def tokenise(text: str):
        return text.lower().split()
    
if __name__ == "__main__":
    builder = DatasetBuilder()
    dataset = builder.run()

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"] 

    retriever = Bm25Retriever(corpus, queries, qrels)

    for query_id, query_text in queries.items():
        recall_at_5, mrr, relevant, retrieved = retriever.evaluate(query_id)
        print(f"Query: {query_text}")
        print(f"Relevant docs: {relevant}")
        print(f"Top-5 retrived: {retrieved}")
        print(f"Num relevant: {len(relevant)}")
        print(f"Recall@5: {recall_at_5}")
        print(f"MRR: {mrr}")