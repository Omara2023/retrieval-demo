Retrieval Benchmark: BM25 vs Dense vs Hybrid (NFCorpus Subset)
0. Overview

This project implements and evaluates three retrieval configurations over a biomedical corpus:

BM25 (sparse lexical retrieval)
Dense retrieval (Sentence-BERT embeddings)
Hybrid retrieval (linear fusion of BM25 + dense scores)

The goal is to compare lexical vs semantic vs hybrid retrieval under realistic biomedical search conditions, where both exact terminology matching and conceptual similarity matter.

1. Why this project / assignment choice

I chose Assignment 1: Retrieval, Honest Comparison because it most directly matches the kind of systems I’ve previously worked with in practice (RAG pipelines using managed embeddings and retrieval APIs), but forces a lower-level implementation where retrieval behavior is visible and measurable.

In prior work, I have used high-level retrieval systems (e.g. Vertex AI / managed vector search) where the retrieval layer is abstracted away. This assignment provided a useful opportunity to:

understand the trade-offs between lexical and semantic retrieval
implement BM25, dense embeddings, and hybrid scoring from first principles
evaluate retrieval behavior under controlled constraints rather than black-box tooling

I also selected this assignment over the Agent / Context Compression tracks because it provides a clearer evaluation surface: retrieval quality can be directly measured using recall and MRR, rather than being confounded by multi-step agent behaviour or summarisation quality.

2. Dataset Choice

I selected NFCorpus because biomedical retrieval creates a natural tension between:

Lexical matching (e.g. drug names, biochemical terms)
Semantic matching (e.g. “immune suppression” vs “reduced leukocyte function”)

This makes it well-suited for comparing BM25, dense retrieval, and hybrid approaches in a controlled but realistic setting.

The corpus was filtered to ~300 documents while preserving all relevant documents for the selected evaluation queries, plus additional randomly sampled distractors.

3. Query Set Design

The evaluation set consists of:

15 selected NFCorpus benchmark queries
5 custom hand-written queries

Each query is manually mapped to ground-truth relevant document IDs.

The custom queries include:

paraphrased queries
multi-document queries
ambiguous / multi-hop queries

This ensures evaluation stress-tests both lexical and semantic retrieval behaviour rather than relying only on keyword overlap.

4. Relevance Judgement Handling

NFCorpus provides graded relevance:

2 = highly relevant
1 = partially relevant

For evaluation consistency with Recall@5 and MRR, relevance was binarised as:

relevance ≥ 1 → relevant

This preserves retrieval signal while keeping evaluation metrics consistent and interpretable.

A potential extension would be evaluating only highly-relevant documents (score = 2), which tends to favour lexical methods like BM25 more strongly.

5. Retrieval Methods
BM25
-Token-based lexical retrieval
-Strong for exact terminology matching
-No semantic understanding

Dense Retrieval
-Sentence-BERT (all-MiniLM-L6-v2)
-Captures semantic similarity
-Slower due to embedding computation

Hybrid Retrieval
-Linear interpolation:
-0.5 * BM25 score
-0.5 * dense cosine similarity
-Designed to balance lexical precision and semantic recall

Evaluation Protocol

All models are evaluated on identical queries using:

Recall@5
Mean Reciprocal Rank (MRR)
p95 latency (ms)

Latency is measured per query during retrieval only, excluding evaluation logic.

7. Results
Retrieval Performance (averaged over repeated runs)
Model	Recall@5	MRR	p95 latency (ms)
BM25	0.297	0.595	0.59
Dense	0.344	0.697	6.20
Hybrid	0.331	0.706	7.35

Notes on stability:
Small run-to-run variation was observed, primarily in MRR for dense vs hybrid configurations. This is expected given:
-stochasticity in evaluation corpus construction (random distractor sampling)
-nondeterminism in floating-point similarity computations (dense embeddings + score aggregation)
-small evaluation set size (20 queries total), which increases metric variance

Despite this, the overall trends remain consistent:
-BM25 is consistently fastest by ~10×
-Dense and hybrid consistently outperform BM25 in ranking quality (MRR)
-Hybrid does not strictly dominate dense on all runs, but remains competitive

Given the scale of the benchmark, conclusions are drawn from relative ordering and trade-offs rather than absolute metric stability.

8. Key Findings
-BM25 provides the strongest efficiency characteristics and remains highly competitive for exact biomedical terminology, but underperforms on semantic or paraphrased queries.
-Dense retrieval improves ranking quality (MRR) and recall on paraphrased queries, confirming the benefit of semantic embeddings in biomedical language where conceptual similarity is important.
-Hybrid retrieval generally improves robustness across query types by combining lexical and semantic signals. However, the linear fusion approach does not guarantee dominance over dense retrieval in all cases, likely due to score scale mismatch and sensitivity to normalization.

Overall, results reflect a classic retrieval trade-off: lexical precision (BM25) vs semantic generalisation (dense) vs robustness (hybrid).

9. Failure Cases

The hybrid model still struggles with:

compositional queries requiring multiple constraints
multi-hop reasoning queries
queries where relevance depends on implicit biomedical relationships rather than lexical overlap

In these cases, BM25 over-emphasises keyword overlap while dense retrieval may over-generalise semantically related but non-relevant documents.

This suggests other models may potentially outperform both approaches.

10. What I Would Improve With More Time
Add a cross-encoder reranker on top of hybrid top-k results
Introduce hard negative sampling during evaluation corpus construction
Evaluate Recall@10 and nDCG@10 for better ranking sensitivity
Perform query-type breakdown (lexical vs semantic vs ambiguous)

11. Reproducibility

Run full pipeline:
make setup
make run

12. Design Decisions & Alternatives Considered

During implementation, I considered several alternative design choices:

Corpus construction
Full NFCorpus vs filtered subset
-Rejected full corpus due to computational cost and reduced ability to iterate quickly on retrieval configurations.
-Instead used a ~300 document subset preserving all relevant documents plus sampled negatives.

Pure random negatives vs hard negatives
-Initially considered keyword-based hard negative mining.
-Rejected due to time constraints and risk of introducing bias in a small evaluation set.
-Used uniform random sampling for simplicity and reproducibility.

Retrieval models
-Dense-only retrieval using larger models (e.g. BGE, MPNet)
-Rejected in favour of all-MiniLM-L6-v2 for latency constraints and local execution requirements.
-Learning-to-rank / cross-encoder reranking
-Identified as a strong improvement for final ranking quality.
-Not implemented due to compute and time constraints, but noted as future work.

Hybrid strategy
-Learned or query-dependent fusion weights
-Rejected in favour of fixed 0.5 / 0.5 interpolation for interpretability and simplicity of comparison.
-Late fusion vs score fusion
-Chose score-level fusion to maintain consistency across BM25 and dense pipelines.

Vector Indexing (FAISS / ANN libraries)
-Considered using FAISS for approximate nearest neighbour search to accelerate dense retrieval.
-Rejected because the corpus size (~300 documents) is small enough that exact brute-force cosine similarity remains fast and avoids additional system complexity.
-This keeps the dense retrieval implementation transparent and easier to compare directly against BM25.