import time
from bm25 import Bm25Retriever
from dense import DenseRetriever
from hybrid import HybridRetriever
from dataset_builder import DatasetBuilder
import numpy as np

class Evaluator:
    def benchmark(self, retriever):
        recalls = []
        mrrs = []
        latencies = []

        for qid, query_text in retriever.queries.items():
            start = time.perf_counter()
            retrieved = retriever._query(query_text)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            recall, mrr, _, _ = retriever.evaluate(qid)
            recalls.append(recall)
            mrrs.append(mrr)

        return {
            "mean_recall@5": np.mean(recalls),
            "mean_mrr": np.mean(mrrs),
            "p95_latency_ms": np.percentile(latencies, 95) * 1000
        }

if __name__ == "__main__":
    builder = DatasetBuilder()
    dataset = builder.run()

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"] 

    bm25 = Bm25Retriever(corpus, queries, qrels)
    dense = DenseRetriever(corpus, queries, qrels)
    hybrid = HybridRetriever(corpus, queries, qrels, bm25, dense)

    eval = Evaluator()

    print(eval.benchmark(bm25))
    print(eval.benchmark(dense))
    print(eval.benchmark(hybrid))