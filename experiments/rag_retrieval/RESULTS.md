# Retrieval evaluation

19 answerable queries (8 clinician-style questions, 11 case-style) and 8 off-corpus queries. A hit means the top-k passages include one from an acceptable paper. Small sample: one query is worth 5.3 points, so differences between models are mostly noise. See the docstring of app/rag/eval_retrieval.py for the stated limits.

| model | hit@1 | hit@3 | hit@5 | MRR | lowest answerable top score | highest off-corpus top score | separable |
|---|---|---|---|---|---|---|---|
| sentence-transformers/multi-qa-distilbert-cos-v1 | 0.895 | 0.895 | 0.947 | 0.908 | 0.39 | 0.532 | no |
| BAAI/bge-small-en-v1.5 | 0.895 | 0.947 | 0.947 | 0.921 | 0.694 | 0.698 | no |
| pritamdeka/S-PubMedBert-MS-MARCO | 0.842 | 0.842 | 0.947 | 0.873 | 0.92 | 0.896 | yes |
| NeuML/pubmedbert-base-embeddings | 0.895 | 0.947 | 0.947 | 0.912 | 0.409 | 0.276 | yes |

Default in app/rag/index.py: `NeuML/pubmedbert-base-embeddings`, minimum passage score 0.34. For the default model the floor 0.34 sits inside the gap between 0.276 and 0.409.
hit@1 ranges from 0.842 to 0.895 across the models, a difference of 1 of 19 queries, which is within noise. Models whose lowest answerable score is above their highest off-corpus score: pritamdeka/S-PubMedBert-MS-MARCO, NeuML/pubmedbert-base-embeddings. Widest margin: NeuML/pubmedbert-base-embeddings (+0.133). Every value depends on the chunking in force when this ran (see app/rag/chunker.py); experiments/rag_retrieval/CHUNKING.md compares chunking strategies.