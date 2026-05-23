import json
import csv
import random
from collections import defaultdict
from pathlib import Path

QUERY_IDS = [4, 8, 88, 98, 282, 325, 349, 408, 518, 572, 573, 665, 1681, 2066, 2961]

TARGET_SIZE = 300

CORPUS_PATH = "nfcorpus/corpus.jsonl" 
QUERIES_PATH = "nfcorpus/queries.jsonl"
Q_RELS_PATH = "./nfcorpus/qrels/train.tsv"

class DatasetBuilder:
    """
    Builds a bounded retrieval benchmark from NFCorpus.

    Output:
    - filtered corpus (~300 docs)
    - selected queries
    - query -> relevant document mappings
    """

    def __init__(self, target_size: int = TARGET_SIZE, 
                 query_ids: list[int] = QUERY_IDS, 
                 corpus_path: str = CORPUS_PATH, 
                 queries_path: str = QUERIES_PATH, 
                 qrels_path: str = Q_RELS_PATH):
        self.target_size = target_size
        self.query_ids = {f"PLAIN-{qid}" for qid in query_ids}
        
        self.corpus_path = Path(corpus_path)
        self.queries_path = Path(queries_path)
        self.qrels_path = Path(qrels_path)

    def run(self):
        """
        Main pipeline. Returns:
        {"corpus": {doc_id: text}, "queries": {query_id: text}, "qrels": {query_id: [doc_ids]}}
        """
        random.seed(42)

        print("[DatasetBuilder] Loading raw data...")

        corpus = self._load_corpus()
        queries = self._load_selected_queries()
        qrels = self._load_qrels()

        print("[DatasetBuilder] Loading synthetic queries...")

        synth_queries, synth_qrels = self._load_custom_queries("custom_qrels.json")

        queries.update(synth_queries)
        qrels.update(synth_qrels)

        print("[DatasetBuilder] Constructing benchmark corpus...")

        filtered_corpus = self._build_filtered_corpus(corpus, qrels)
        
        print(f"[DatasetBuilder] Final corpus contains {len(filtered_corpus)} documents.")

        return {"corpus": filtered_corpus, "queries": queries, "qrels": qrels}

    def _load_corpus(self):
        """Loads full corpus."""
        self._validate_path(self.corpus_path)

        corpus = {}
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = {
                    "title": doc.get("title", ""),
                    "text": doc["text"]
                }

        print(f"[Corpus] Loaded {len(corpus)} documents.")
        return corpus
    
    def _load_selected_queries(self) -> dict[str, str]:
        """
        Loads only selected benchmark queries. Returns: {query_id: query_text}
        """
        self._validate_path(self.queries_path)

        queries = {}

        with open(self.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line)
                
                query_id = query["_id"]
                if query_id in self.query_ids:
                    queries[query_id] = query["text"]

        print(f"[Queries] Loaded {len(queries)} selected queries.")

        return queries

    def _load_qrels(self) -> dict[str, list[str]]:
        """
        Loads query -> relevant document mappings.

        NFCorpus uses graded relevance:
        - 2 = highly relevant
        - 1 = partially relevant

        For this assignment we treat score >= 1 as relevant.

        Returns: {query_id: [relevant_doc_ids]}
        """

        self._validate_path(self.qrels_path)
        mapping = defaultdict(list)

        with open(self.qrels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                query_id = row["query-id"]
                if query_id not in self.query_ids:
                    continue
                relevance = int(row["score"])

                if relevance >= 1:
                    mapping[query_id].append(row["corpus-id"])

        print(f"[Qrels] Loaded mappings for {len(mapping)} queries.")

        return dict(mapping)

    def _build_filtered_corpus(self, corpus: dict[str, str], qrels: dict[str, list[str]]) -> dict[str, dict[str, str]]:
        """
        Constructs bounded evaluation corpus.

        Strategy:
        1. Include ALL relevant documents
        2. Sample random distractors from remaining corpus
        3. Return final filtered corpus
        """

        positive_doc_ids: set[str] = set()

        for docs in qrels.values():
            positive_doc_ids.update(docs)

        print(f"[CorpusBuilder] Found {len(positive_doc_ids)} positive documents.")

        all_doc_ids = set(corpus.keys())
        negative_pool = list(all_doc_ids - positive_doc_ids)
        required_negatives = max(0, self.target_size - len(positive_doc_ids))
        distractors = random.sample(negative_pool, required_negatives)

        final_doc_ids = positive_doc_ids | set(distractors)

        filtered_corpus = {
            doc_id: corpus[doc_id]
            for doc_id in final_doc_ids
        }
        
        return filtered_corpus
        
    @staticmethod
    def _validate_path(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"{path} could not be found.")

    def _load_custom_queries(self, path="custom_qrels.json"):
        self._validate_path(Path(path))

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = {}
        qrels = {}

        for i, item in enumerate(data):
            qid = f"SYNTH-{i}"
            queries[qid] = item["query"]
            qrels[qid] = item["ground_truth_ids"]

        return queries, qrels

    def write_corpus_to_file(
        self,
        corpus: dict[str, dict[str, str]],
        path: str = "output.csv"
    ):
        """Writes corpus dict {doc_id: {title, text}} to CSV file."""

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["doc_id", "title", "text"])

            for doc_id, data in corpus.items():
                writer.writerow([
                    doc_id,
                    data.get("title", ""),
                    data.get("text", "")
                ])

        print(f"[Corpus] Wrote {len(corpus)} documents to {path}")

if __name__ == "__main__":
    builder = DatasetBuilder() 
    dataset = builder.run()

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]  

    #builder.write_corpus_to_file(corpus)

    for i in corpus:
        print(corpus[i])
        break

    print("[Done] Dataset successfully built.")

