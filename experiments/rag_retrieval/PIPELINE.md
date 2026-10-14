# Retrieval pipeline comparison (stage 2: 27 papers, 474 passages)

54 answerable queries split by topic group into dev and test, plus 16 off-corpus queries. Passage relevance is a keyword proxy (see eval_queries.py), audited by hand in experiments/rag_retrieval/AUDIT.md. One query is about 4 points on a half, so gaps of one or two queries are noise. Random ranking would score p@1 0.051, p@3 0.142, p@5 0.22.

| variant | dev p@1 | dev p@3 | dev MRR | dev nDCG@5 | test p@1 | test p@3 | test MRR | test nDCG@5 | abstention AUC (dense score) | abstention AUC (reranker score) |
|---|---|---|---|---|---|---|---|---|---|---|
| dense (baseline) | 0.731 | 0.962 | 0.849 | 0.714 | 0.571 | 0.821 | 0.707 | 0.627 | 0.969 | - |
| keyword only (BM25) | 0.769 | 1.0 | 0.865 | 0.758 | 0.714 | 0.964 | 0.815 | 0.708 | 0.955 | - |
| dense + expansion | 0.692 | 0.923 | 0.809 | 0.716 | 0.607 | 0.821 | 0.725 | 0.653 | 0.97 | - |
| hybrid | 0.962 | 1.0 | 0.981 | 0.911 | 0.714 | 1.0 | 0.851 | 0.762 | 0.966 | - |
| hybrid + expansion | 0.885 | 0.923 | 0.914 | 0.87 | 0.75 | 0.893 | 0.815 | 0.717 | 0.965 | - |
| hybrid + expansion + MiniLM rerank | 0.962 | 1.0 | 0.981 | 0.884 | 0.893 | 0.929 | 0.911 | 0.832 | 0.966 | 0.953 |
| hybrid + expansion + MedCPT rerank | 0.923 | 1.0 | 0.962 | 0.911 | 0.821 | 0.964 | 0.893 | 0.839 | 0.965 | 0.997 |
| hybrid + MedCPT rerank | 0.923 | 1.0 | 0.962 | 0.911 | 0.857 | 0.929 | 0.902 | 0.844 | 0.963 | 0.997 |
| hybrid + MedCPT rerank (top 10) | 1.0 | 1.0 | 1.0 | 0.961 | 0.857 | 0.929 | 0.893 | 0.829 | 0.964 | 0.997 |
| hybrid + MedCPT rerank (top 5) | 1.0 | 1.0 | 1.0 | 0.971 | 0.893 | 0.964 | 0.938 | 0.873 | 0.966 | 0.997 |