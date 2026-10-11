# Chunking strategy comparison

19 answerable queries, 8 off-corpus. passage = right paper AND the question's key terms present. One query is worth about 5 points, so most gaps are noise. ctx3 = words the explanation LLM would be shown for the top 3. gap = lowest answerable top score minus highest off-corpus top score (positive means a score threshold can refuse off-corpus queries).

| model | strategy | chunks | median words | paper@1 | passage@1 | passage@3 | passage MRR | ctx3 | gap |
|---|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | sentence 100w | 592 | 91 | 0.895 | 0.789 | 0.895 | 0.851 | 279 | -0.019 |
| bge-small-en-v1.5 | sentence 170w, 1-sentence overlap | 352 | 146 | 0.842 | 0.737 | 0.947 | 0.833 | 400 | -0.008 |
| bge-small-en-v1.5 | sentence 300w + ~15% overlap | 238 | 187 | 0.895 | 0.737 | 0.947 | 0.833 | 602 | -0.004 |
| bge-small-en-v1.5 | whole section (truncated by model) | 177 | 167 | 0.895 | 0.789 | 0.947 | 0.86 | 765 | -0.019 |
| bge-small-en-v1.5 | semantic | 326 | 105 | 0.895 | 0.737 | 0.895 | 0.826 | 361 | -0.006 |
| pubmedbert-base-embeddings | sentence 100w | 592 | 91 | 0.842 | 0.632 | 0.842 | 0.747 | 277 | +0.193 |
| pubmedbert-base-embeddings | sentence 170w, 1-sentence overlap | 352 | 146 | 0.842 | 0.579 | 0.842 | 0.712 | 422 | +0.122 |
| pubmedbert-base-embeddings | sentence 300w + ~15% overlap | 238 | 187 | 0.895 | 0.737 | 0.895 | 0.818 | 641 | +0.143 |
| pubmedbert-base-embeddings | whole section (truncated by model) | 177 | 167 | 0.842 | 0.737 | 0.947 | 0.833 | 794 | +0.149 |
| pubmedbert-base-embeddings | semantic | 323 | 102 | 0.842 | 0.632 | 0.895 | 0.762 | 405 | +0.132 |