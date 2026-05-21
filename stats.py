import csv
from pathlib import Path

class Loader:

    def __init__(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Path: {path} does not exist.")
        self.path = path

    def _load_tsv(self):
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            return list(reader)

    def produce_counts(self):
        records = self._load_tsv()
        counts = {}

        for record in records:
            id = record["query-id"]
            counts[id] = counts.get(id, 0) + 1

        return counts
    
    def produce_targeted_counts(self, targets: list):
        counts = {}
        if len(targets) == 0:
            print("Empty target list.")
            return counts
        
        records = self._load_tsv()
        
        for record in records:
            id = record["query-id"]

            if id not in targets:
                continue

            counts[id] = counts.get(id, 0) + 1

        return counts


if __name__ == "__main__":
    loader = Loader("./nfcorpus/qrels/train.tsv")

    raw_ids = [4, 8, 88, 98, 282, 325, 349, 408, 518, 572, 573, 665, 1681, 2066, 2961]

    targets = [f"PLAIN-{i}" for i in raw_ids]
    counts = loader.produce_targeted_counts(targets)

    for i, v in counts.items():
        print(i, v)

    print(f"Selected documents: {sum([v for v in counts.values()])}")