"""
Embeds the INDICATION text of the Excluded studies with the same frozen Bio_ClinicalBERT and the same pooling as scripts/train_text_baseline.py, so the
text model can be trained and scored on the fourth class (experiments/v2_four_class/PLAN.md, step 2). The in-scope embeddings already exist.

Texts are sorted by length before batching: the median indication is 8 words and a few run past 100, and padding every batch of 32 to its longest text
wastes most of the work. Mean pooling ignores padding, so this changes the speed and not the result; the script proves that on in-scope studies whose
embeddings are cached before it starts. Chunked and resumable. Only the three roles of the plan can be asked for.

Output: experiments/v2_four_class/emb_indication_excluded_<role>.npy, rows in the order of data/processed/excluded_<role>.csv (git-ignored).
Run from the repo root:  python scripts/v2_embed_excluded.py
"""
import torch  # noqa: F401  (Windows: torch before sklearn, see CLAUDE.md gotcha 7)

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import train_text_baseline as tb  # noqa: E402

OUT = ROOT / "experiments" / "v2_four_class"
CHUNKS = OUT / "_emb_chunks"
ROLES = ("train", "val", "none")
CHUNK = 640
PROOF = 96                                            # cached in-scope validation embeddings recomputed in sorted batches


def indications(paths) -> list[str]:
    return [tb.extract_indication(tb.read_report(p)) for p in paths]


def embed_sorted(texts: list[str], tok, model) -> np.ndarray:
    order = np.argsort([len(t.split()) for t in texts], kind="stable")
    emb = tb.embed([texts[i] for i in order], tok, model)
    out = np.zeros_like(emb)
    out[order] = emb
    return out


def prove_sorting_changes_nothing(tok, model) -> float:
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv").head(PROOF)
    cached = np.load(ROOT / "experiments" / "text_baseline" / "emb_indication_val.npy")[:PROOF]
    fresh = embed_sorted(indications(val.report_path), tok, model)
    return float(np.abs(fresh - cached).max())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from transformers import AutoModel, AutoTokenizer
    CHUNKS.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tb.MODEL_NAME)
    model = AutoModel.from_pretrained(tb.MODEL_NAME).eval()
    diff = prove_sorting_changes_nothing(tok, model)
    print(f"largest difference from the cached in-scope embeddings, {PROOF} studies in sorted batches: {diff:.2e}", flush=True)
    assert diff < 1e-3, "sorted batching changed the embeddings"

    rows = []                                                                    # (role, row number, text)
    for role in ROLES:
        table = pd.read_csv(ROOT / "data" / "processed" / f"excluded_{role}.csv")
        rows += [(role, i, t) for i, t in enumerate(indications(table.report_path))]
    order = np.argsort([len(t.split()) for _, _, t in rows], kind="stable")
    n_chunks = -(-len(order) // CHUNK)
    t0 = time.time()
    for k in range(n_chunks):
        f = CHUNKS / f"c_{k:03d}.npy"
        if f.exists():
            continue
        idx = order[k * CHUNK:(k + 1) * CHUNK]
        np.save(f, tb.embed([rows[i][2] for i in idx], tok, model))
        print(f"chunk {k + 1}/{n_chunks} done, {time.time() - t0:.0f}s", flush=True)
    emb = np.vstack([np.load(CHUNKS / f"c_{k:03d}.npy") for k in range(n_chunks)])
    assert len(emb) == len(rows)
    by_role = {r: np.zeros((sum(1 for x in rows if x[0] == r), emb.shape[1]), np.float32) for r in ROLES}
    for pos, i in enumerate(order):
        role, row, _ = rows[i]
        by_role[role][row] = emb[pos]
    for role, arr in by_role.items():
        np.save(OUT / f"emb_indication_excluded_{role}.npy", arr)
        print(f"saved {role}: {arr.shape}")


if __name__ == "__main__":
    main()
