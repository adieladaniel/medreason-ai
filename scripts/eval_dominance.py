"""
Does the text input carry the project, and what do the image and blood models add? Validation split only (the
test split stays sealed); prints aggregate numbers and no patient text.

Why this exists: on the Analyze page the text model decided most cases, which raises the fair question of whether
the image and blood models are worth having. This answers it with the saved validation predictions:

1. What each input adds when the other two are already there, with 95 percent intervals that resample whole
   patients (one patient has several studies).
2. How much of the text model is plain keywords: a crude rule written from clinical common sense, not tuned on
   these cases, against the trained text model.
3. The cases whose indication has no cancer or infection word, where text has little to go on.
4. The true Lung Cancer cases one by one: which model finds which.
5. Who decides the fused answer, and per-class scores.
6. A cancer alert as an alternative to "highest probability wins": the threshold is picked on the training
   out-of-fold predictions and applied once to validation.

Limits: the keyword lists and the "no clue" subset are mine and have not been checked by anyone else; 117 cancer cases
make every cancer number wide; part 6 is one run and a design option, not a result to quote as final.

Run from the repo root:  python scripts/eval_dominance.py
"""
import torch  # noqa: F401  (Windows: torch before sklearn, see CLAUDE.md gotcha 7)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from explain.text import extract_indication  # noqa: E402
from scripts.train_fusion import assemble  # noqa: E402

rng = np.random.default_rng(7)
MOD = {"image": 0, "blood": 1, "text": 2}              # position of each model's block in the 9 fusion features
CANCER_WORDS = ["metasta", "malignan", "carcinoma", "cancer", "mass", "nodule", "neoplasm", "tumor", "tumour", "lesion",
                "lymphoma", "nsclc", "oncolog", "chemo"]
INFECTION_WORDS = ["pneumonia", "infiltrat", "consolidat", "cough", "fever", "sepsis", "infect", "aspirat", "febrile",
                   "leukocytosis", "dyspnea", "shortness of breath", "hypox", "sob"]
CONFIGS = {"text": ["text"], "blood": ["blood"], "image": ["image"], "image+blood": ["image", "blood"],
           "blood+text": ["blood", "text"], "image+text": ["image", "text"], "all three": ["image", "blood", "text"]}


def pooled(P, mods):
    """Normalised geometric mean of the chosen models' probabilities (the belief state's rule)."""
    s = P[:, [MOD[m] for m in mods], :].clip(1e-6)
    g = np.exp(np.log(s).mean(axis=1))
    return g / g.sum(axis=1, keepdims=True)


def macro_f1(y, pred, idx=slice(None)):
    return f1_score(y[idx], pred[idx], average="macro", labels=[0, 1, 2], zero_division=0)


def main() -> None:
    merged, X, y = assemble("val")
    P = X.reshape(len(X), 3, 3)                        # [case, model (image, blood, text), class]
    val = pd.read_csv("data/processed/val.csv").set_index("study_id").loc[merged["study_id"].values]
    subject = val["subject_id"].values
    pred = {k: pooled(P, v).argmax(axis=1) for k, v in CONFIGS.items()}
    print(f"validation cases: {len(y)}  (Normal {np.sum(y == 0)}, Pneumonia {np.sum(y == 1)}, Lung Cancer {np.sum(y == 2)})\n")

    # 1. what each input adds
    by_subject = {s: np.where(subject == s)[0] for s in np.unique(subject)}
    keys = list(by_subject)
    boot = {k: [] for k in pred}
    for _ in range(1000):
        idx = np.concatenate([by_subject[s] for s in rng.choice(keys, len(keys))])
        for k in pred:
            boot[k].append(macro_f1(y, pred[k], idx))
    boot = {k: np.array(v) for k, v in boot.items()}
    print("1. macro F1 by which inputs are used (95% interval over patients)")
    for k in CONFIGS:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        print(f"   {k:<12} {macro_f1(y, pred[k]):.3f}  ({lo:.3f} to {hi:.3f})")
    print("\n   what each input adds when the other two are already there (difference in macro F1, 95% interval):")
    for name, base in (("image", "blood+text"), ("blood", "image+text"), ("text", "image+blood")):
        d = boot["all three"] - boot[base]
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "clearly above zero" if lo > 0 else "cannot be told from zero"
        print(f"   adding {name:<6} to {base:<11}: {macro_f1(y, pred['all three']) - macro_f1(y, pred[base]):+.3f}  ({lo:+.3f} to {hi:+.3f})  {verdict}")

    # 2. how much of the text model is plain keywords
    texts = [extract_indication(Path(p).read_text(encoding="utf-8", errors="ignore")).lower() for p in val["report_path"]]
    hit_c = np.array([any(w in t for w in CANCER_WORDS) for t in texts])
    hit_p = np.array([any(w in t for w in INFECTION_WORDS) for t in texts])
    rule = np.where(hit_c, 2, np.where(hit_p, 1, 0))
    print("\n2. a crude keyword rule on the indication (written from clinical common sense, not tuned on these cases)")
    print("   rule: cancer-type word -> Lung Cancer, else infection-type word -> Pneumonia, else Normal")
    print(f"   keyword rule macro F1 {macro_f1(y, rule):.3f}   |   the trained text model {macro_f1(y, pred['text']):.3f}   |   all three models {macro_f1(y, pred['all three']):.3f}")
    print(f"   true Lung Cancer cases with a cancer-type word in the indication: {hit_c[y == 2].mean():.0%}; true Pneumonia with an infection-type word: {hit_p[y == 1].mean():.0%}; Normal with any such word: {(hit_c | hit_p)[y == 0].mean():.0%}")
    print(f"   cases whose indication has none of these words: {(~(hit_c | hit_p)).mean():.0%}")

    # 3. cases where the indication gives no clue
    quiet = ~(hit_c | hit_p)
    print(f"\n3. the {quiet.sum()} cases whose indication has no cancer or infection word (Normal {np.sum(y[quiet] == 0)}, Pneumonia {np.sum(y[quiet] == 1)}, Lung Cancer {np.sum(y[quiet] == 2)})")
    for k in ("text", "blood", "image", "image+blood", "blood+text", "all three"):
        print(f"   {k:<12} macro F1 {macro_f1(y, pred[k], quiet):.3f}   accuracy {np.mean(pred[k][quiet] == y[quiet]):.0%}")

    # 4. the cancer cases one by one
    c = np.where(y == 2)[0]
    ok = {k: pred[k][c] == 2 for k in ("text", "blood", "image", "all three")}
    print(f"\n4. the {len(c)} true Lung Cancer cases: how many each model gets right on its own")
    for k, v in ok.items():
        print(f"   {k:<10} {v.sum():>3} ({v.mean():.0%})")

    def only(a, others):
        return int(np.sum(ok[a] & ~np.any([ok[o] for o in others], axis=0)))
    print(f"   found ONLY by text: {only('text', ['blood', 'image'])}; ONLY by blood: {only('blood', ['text', 'image'])}; ONLY by image: {only('image', ['text', 'blood'])}; by none of the three: {int(np.sum(~ok['text'] & ~ok['blood'] & ~ok['image']))}")
    print(f"   text found it and the fused answer lost it: {int(np.sum(ok['text'] & ~ok['all three']))}; text missed it and the fused answer found it: {int(np.sum(~ok['text'] & ok['all three']))}")
    meta = np.array(["metasta" in texts[i] for i in c])
    print(f"   cancer cases with a 'metasta' word in the indication: {meta.mean():.0%}; of those, text finds {ok['text'][meta].mean():.0%}")

    # 5. who decides
    conf = P.max(axis=2)
    top = conf.argmax(axis=1)
    print("\n5. who decides")
    print(f"   fused class equals the text model's class in {np.mean(pred['all three'] == pred['text']):.0%} of cases, the blood model's in {np.mean(pred['all three'] == pred['blood']):.0%}, the image model's in {np.mean(pred['all three'] == pred['image']):.0%}")
    print(f"   most confident of the three: text {np.mean(top == 2):.0%}, blood {np.mean(top == 1):.0%}, image {np.mean(top == 0):.0%}; average top probability: text {conf[:, 2].mean():.2f}, blood {conf[:, 1].mean():.2f}, image {conf[:, 0].mean():.2f}")
    print("   per-class F1 (Normal / Pneumonia / Lung Cancer):")
    for k in ("image", "blood", "text", "all three"):
        f = f1_score(y, pred[k], average=None, labels=[0, 1, 2], zero_division=0)
        print(f"   {k:<10} {f[0]:.2f} / {f[1]:.2f} / {f[2]:.2f}")

    # 6. a cancer alert instead of "highest probability wins"
    _, Xtr, ytr = assemble("train")
    ptr = pooled(Xtr.reshape(len(Xtr), 3, 3), list(MOD))[:, 2]
    pva = pooled(P, list(MOD))[:, 2]
    base = pooled(P, list(MOD)).argmax(axis=1)
    print("\n6. a cancer alert instead of 'highest probability wins'")
    print(f"   train out-of-fold cases {len(ytr)} ({np.sum(ytr == 2)} cancer); validation {len(y)} ({np.sum(y == 2)} cancer)")
    print(f"   today's answer finds {np.sum((base == 2) & (y == 2))} of {np.sum(y == 2)} cancers ({np.mean(base[y == 2] == 2):.0%}) with {np.sum(base == 2)} cancer calls")
    print("   alert rule: also flag 'possible lung cancer' when the fused cancer probability is at least t;")
    print("   t is picked on train so the alert would catch the target share of the true cancers there")
    print(f"   {'target recall (train)':<22}{'t':>7}  {'validation: cancers caught':>28}  {'alerts per 100 cases':>22}  {'alerts that are cancer':>24}")
    for target in (0.5, 0.6, 0.7, 0.8):
        t = np.quantile(ptr[ytr == 2], 1 - target)
        a = pva >= t
        caught = np.sum(a & (y == 2))
        print(f"   {target:<22.0%}{t:>7.3f}  {caught:>3} of {np.sum(y == 2)} ({caught / np.sum(y == 2):>4.0%}){'':>10}  {100 * a.mean():>20.1f}  {caught / max(a.sum(), 1):>23.0%}")
    print(f"   for scale: cancer is {np.mean(y == 2):.1%} of validation cases, so an alert that fired on everything would be right {np.mean(y == 2):.1%} of the time")


if __name__ == "__main__":
    main()
