import json
import csv
import random
from pathlib import Path

QUERY_IDS = [4, 8, 88, 98, 282, 325, 349, 408, 518, 572, 573, 665, 1681, 2066, 2961]
TARGET_SIZE = 300
CORPUS_PATH = "nfcorpus/corpus.jsonl" 
QUERIES_PATH = "nfcorpus/queries.jsonl"
MAPPINGS_PATH = "./nfcorpus/qrels/train.tsv"

class DatasetBuilder:

    def __init__(self, target_size: int, query_ids: list[int], corpus_path: str, queries_path: str, query_mapping_path: str):
        self.target_size = target_size
        self.query_ids = [f"PLAIN-{i}" for i in query_ids]
        self.corpus_path = corpus_path
        self.queries_path = queries_path
        self.query_mapping_path = query_mapping_path

    def run(self):
        """Return filtered corpus, query to answer mapping and 15 queries."""
        random.seed(42)
        corpus = self._generate_filtered_corpus()
        query_mappings = self._get_query_document_ids()
        queries = self._load_selected_queries()

        return corpus, query_mappings, queries

    def _generate_filtered_corpus(self) -> dict:
        positive_doc_ids = set()

        output = self._get_query_document_ids()
        for docs in output.values():
            positive_doc_ids.update(docs)

        all_doc_ids = set(self._load_corpus().keys())
        negative_doc_ids = all_doc_ids - positive_doc_ids
        distractors = random.sample(list(negative_doc_ids), self.target_size - len(positive_doc_ids))

        final_doc_ids = positive_doc_ids | set(distractors)

        filtered_corpus = {}
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                if doc["_id"] in final_doc_ids:
                    filtered_corpus[doc["_id"]] = doc["text"] 
        return filtered_corpus

    def _load_corpus(self):
        corpus = {}
        if not Path(self.corpus_path).exists():
            raise FileNotFoundError(f"Corpus {self.corpus_path} could not be found.")
        
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = doc["text"]

        return corpus

    def _load_selected_queries(self):
        if not Path(self.queries_path).exists():
            raise FileNotFoundError(f"Queries {self.queries_path} could not be found.")
        
        queries = {}
        with open(self.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line)
                id = query["_id"]
                if id in self.query_ids:
                    queries[id] = query["text"]

        print(f"[Harness] Loaded {len(queries)} queries.") 
        return queries
    
    def _get_query_document_ids(self):
        """Returns a dict of query_id : list[corpus_id]"""
        mapping = {}
        if len(self.query_ids) == 0:
            raise Exception("Empty target list.")
            
        records = self._load_tsv()
        
        for record in records:
            id = record["query-id"]

            if id not in self.query_ids:
                continue

            mapping[id] = mapping.get(id, []) + [record["corpus-id"]]

        return mapping
    
    def _load_tsv(self):
        with open(self.query_mapping_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            return list(reader)  

if __name__ == "__main__":
    builder = DatasetBuilder(TARGET_SIZE, QUERY_IDS, CORPUS_PATH, QUERIES_PATH, MAPPINGS_PATH) 
    c, m, q = builder.run()    

