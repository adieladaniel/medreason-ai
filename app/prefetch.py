"""
Downloads every pretrained model the app needs, once, into the folders the app reads them from.

Why this exists: the container runs with HF_HUB_OFFLINE=1, so a clinical-text service never reaches out to the
internet at request time and a missing model fails loudly at startup instead of being fetched silently. The models
therefore have to be put in place deliberately, into the volumes compose mounts (the Hugging Face cache and the
torchxrayvision folder). This loads each one through the same function the app uses, so what is downloaded is exactly
what is read later, and nothing else (the local Hugging Face cache on this laptop holds 2.7 GB, much of it models
that were tried and rejected).

Run with the network on:
    docker compose run --rm -e HF_HUB_OFFLINE=0 setup python -m app.prefetch
It needs the trained models mounted (experiments/), because the image loader reads the classifier head from there.
"""
from __future__ import annotations

import torch  # noqa: F401  (before sklearn and shap on Windows, see CLAUDE.md gotcha 7)

import time


def downloads() -> dict:
    """Name -> loader for each model that is fetched from the internet the first time it is used."""
    import explain.image as image
    import explain.text as text
    from app.rag import faithfulness, hybrid, index, retrieve
    return {
        "image backbone (torchxrayvision)": image._load,
        "text encoder (Bio_ClinicalBERT)": text._load,
        "passage embedder": lambda: index._model(index.DEFAULT_MODEL),
        "reranker": lambda: hybrid._reranker(retrieve.RERANKER),
        "entailment checker": faithfulness._model,
    }


def main() -> None:
    for name, load in downloads().items():
        t0 = time.perf_counter()
        load()
        print(f"{name}: ready in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
