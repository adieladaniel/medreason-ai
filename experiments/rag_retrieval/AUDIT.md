# Hand audit of the keyword relevance proxy

Every retrieval number in PIPELINE.md counts a passage as relevant when it comes from an allowed
topic group and contains the question's key terms (app/rag/eval_queries.py). That is a proxy, so
18 top-1 passages from the chosen pipeline (hybrid search plus the MedCPT reranker) were read
against their queries and judged by hand: all 6 where the proxy said "not relevant", and 12
sampled at random (seed 7) where it said "relevant". Judged by the project author's assistant,
not an independent annotator, and 18 is a small sample.

| # | proxy | judgment | note |
|---|---|---|---|
| 1 | not relevant | relevant | procalcitonin and bacterial pneumonia: general abstract on PCT, CRP, blood count. The proxy was too strict |
| 2 | not relevant | not relevant | WBC and consolidation: a case report with a normal blood count. Reranker score -1.90, so the evidence threshold refuses it. Correct behavior |
| 3 | not relevant | not relevant | platelet-lymphocyte ratio in pneumonia: top result was an NLR passage (reranker score 15.7), while a paper on exactly this is in the corpus. A real ranking error that the threshold cannot catch |
| 4 | not relevant | partly relevant | routine blood tests and pneumonia outcomes: abstract naming CRP, PCT, blood count. A better source (PMC12890234) ranked lower |
| 5 | not relevant | weak | what lung cancer looks like on X-ray: abstract of a missed-cancers review, not a description |
| 6 | not relevant | partly relevant | nodules over 3 cm: nodule-management abstract with no statement about size |
| 7 | relevant | clearly relevant | thrombocytosis in lung cancer |
| 8 | relevant | on topic, weak | LDH in lung cancer: a methods paragraph that states no finding |
| 9 | relevant | clearly relevant | CURB-65 and PSI mortality |
| 10 | relevant | clearly relevant | albumin and pneumonia outcomes |
| 11 | relevant | clearly relevant | lymphopenia and pneumonia outcomes |
| 12 | relevant | on topic, weak | malignant effusion imaging: a mesothelioma passage |
| 13 | relevant | clearly relevant | low lymphocyte count in pneumonia |
| 14 | relevant | clearly relevant | low albumin in pneumonia |
| 15 | relevant | clearly relevant | spiculated nodule and cancer risk |
| 16 | relevant | clearly relevant | NLR in non-small cell lung cancer |
| 17 | relevant | on topic, weak | anemia in lung cancer: the part shown is a generic lung cancer introduction |
| 18 | relevant | clearly relevant | HALP score in lung cancer |

## What it shows

- Of 12 proxy "relevant" passages, 9 clearly answered the question, 3 were on topic but weak, and
  none were wrong. The proxy does not invent relevance, but it counts weak passages as hits.
- Of 6 proxy "not relevant" passages, 3 were actually relevant or partly relevant, 1 was weak,
  and 2 were real errors (one of them correctly refused by the threshold). The proxy undercounts.
- The two error directions partly cancel, so absolute levels should be read as approximate.
  Comparisons between pipelines are safer, since the same proxy scores all of them.
- A high reranker score means the passage is on the query's topic, not that it answers the
  question (case 3). The explanation writer has to check what a passage actually states.
