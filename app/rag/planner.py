"""
Evidence planner: turns what the agent already knows about one case into a few retrieval
queries, one per real driver, in the terse keyword style that app/rag/eval_pipeline.py
evaluated ("lung cancer with elevated LDH").

Why per-driver queries instead of one query built from the label: a query of just "lung
cancer" retrieves generic cancer text, and one mega-query blurs the specific facts. A query
per driver also lets the explanation attach each piece of evidence to the finding it
supports.

What counts as a driver:
  blood   a lab SHAP says pushed toward the predicted class AND whose value is outside its
          adult reference range in a direction we have a phrase for. A supporting feature
          with a normal value gets no query (there is no literature on "normal albumin"
          supporting cancer), and is listed under `skipped` so nothing is hidden.
  image   the top pathology channels that supported the prediction with a clearly present
          signal (value >= 0.5).
  text    clinically meaningful words from the indication that supported the prediction
          ("mass", "biopsy"), from a fixed vocabulary so filler words never become queries.
  conflict  when modalities disagree and one of them predicted the other disease, one
          differential query ("pneumonia versus lung cancer"), because that is exactly when a
          reader needs literature on telling them apart.

A modality whose own prediction differs from the final one is not used for driver queries:
its attributions explain a different class. That is recorded under `skipped`.

Design rules kept from the project: this module only reads the agent's output and writes
queries. It never sees or changes the prediction, and for a Normal prediction it asks for
nothing (there are no abnormal drivers to explain). Reference ranges are approximate adult
intervals in the units the dataset uses, written from general clinical knowledge, not
derived from the data; they decide which values are worth querying, not what is diagnostic.
"""
from __future__ import annotations

MAX_QUERIES = 5
IMAGE_MIN_VALUE = 0.5

CLASS_PHRASE = {"Pneumonia": "pneumonia", "Lung Cancer": "lung cancer"}

# feature name as emitted by explain/blood.py -> (low limit, high limit, phrase if low, phrase if high)
BLOOD_RANGES: dict[str, tuple[float, float, str | None, str | None]] = {
    "WBC": (4.0, 11.0, "leukopenia", "elevated white blood cell count"),
    "RBC": (4.0, 5.9, "low red blood cell count", None),
    "hemoglobin": (12.0, 17.5, "low hemoglobin", None),
    "hematocrit": (36.0, 52.0, "low hematocrit", None),
    "platelets": (150.0, 400.0, None, "thrombocytosis"),
    "neutrophils %": (40.0, 75.0, None, "neutrophilia"),
    "lymphocytes %": (20.0, 45.0, "low lymphocyte count", None),
    "monocytes %": (2.0, 10.0, None, "elevated monocytes"),
    "CRP": (0.0, 10.0, None, "elevated C-reactive protein"),
    "ESR": (0.0, 20.0, None, "elevated erythrocyte sedimentation rate"),
    "albumin": (3.5, 5.2, "low albumin", None),
    "LDH": (120.0, 250.0, None, "elevated LDH"),
    "sodium": (135.0, 145.0, "hyponatremia", None),
    "potassium": (3.5, 5.2, None, None),
    "glucose": (70.0, 140.0, None, None),
    "creatinine": (0.6, 1.3, None, None),
    "urea": (7.0, 20.0, None, "elevated urea"),
    "neutrophil-lymphocyte ratio": (0.0, 3.0, None, "high neutrophil-lymphocyte ratio"),
    "platelet-lymphocyte ratio": (0.0, 150.0, None, "high platelet-lymphocyte ratio"),
}

IMAGE_PHRASES = {
    "Consolidation": "consolidation", "Lung Opacity": "lung opacity", "Edema": "pulmonary edema",
    "Effusion": "pleural effusion", "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly",
    "Lung Lesion": "lung nodule or mass", "Pneumothorax": "pneumothorax",
    "Enlarged Cardiomediastinum": "enlarged mediastinum",
}


# words from an indication worth a literature query. Anything else ("man", "the") is not.
CLINICAL_TERMS = {"mass", "nodule", "lesion", "tumor", "tumour", "malignancy", "metastasis", "metastases",
                  "carcinoma", "cancer", "biopsy", "hemoptysis", "cough", "fever", "consolidation",
                  "infiltrate", "opacity", "effusion", "atelectasis", "dyspnea", "wheeze", "sepsis",
                  "pneumonia", "aspiration", "edema", "empyema"}
SELF_WORDS = {"Pneumonia": {"pneumonia"}, "Lung Cancer": {"cancer", "carcinoma", "malignancy"}}


# Substrings a background sentence must contain (any one) to count as being about the finding a query was
# written for. Without this a sentence can be faithful to its passage yet about something else: the
# "lung opacity" query pulled a case report of persistent consolidation later found to be cancer, and a
# CT-timing note, and the writer model never answered NOT_ADDRESSED on its own.
TERMS = {
    "leukopenia": ("leukopeni", "leucopeni", "white blood", "wbc"),
    "elevated white blood cell count": ("white blood", "wbc", "leukocyt", "leucocyt"),
    "low red blood cell count": ("red blood", "rbc", "anemi", "anaemi"),
    "low hemoglobin": ("hemoglobin", "haemoglobin", "anemi", "anaemi"),
    "low hematocrit": ("hematocrit", "haematocrit"),
    "thrombocytosis": ("thrombocytosis", "platelet"),
    "neutrophilia": ("neutrophil",),
    "low lymphocyte count": ("lymphocyt", "lymphopeni"),
    "elevated monocytes": ("monocyt",),
    "elevated C-reactive protein": ("c-reactive", "crp"),
    "elevated erythrocyte sedimentation rate": ("sedimentation", "esr"),
    "low albumin": ("albumin",),
    "elevated LDH": ("ldh", "lactate dehydrogenase"),
    "hyponatremia": ("hyponatr",),
    "elevated urea": ("urea", "bun"),
    "high neutrophil-lymphocyte ratio": ("neutrophil-to-lymphocyte", "neutrophil-lymphocyte", "neutrophil to lymphocyte", "nlr"),
    "high platelet-lymphocyte ratio": ("platelet-to-lymphocyte", "platelet-lymphocyte", "platelet to lymphocyte", "plr"),
    "consolidation": ("consolidat",),
    "lung opacity": ("opacit", "infiltrat", "air-space", "airspace", "ground-glass", "ground glass"),
    "pulmonary edema": ("edema", "oedema"),
    "pleural effusion": ("effusion",),
    "atelectasis": ("atelecta",),
    "cardiomegaly": ("cardiomegal", "cardiac enlargement", "enlarged heart"),
    "lung nodule or mass": ("nodule", "mass", "lesion"),
    "pneumothorax": ("pneumothora",),
    "enlarged mediastinum": ("mediastin",),
}


def _blood_phrase(item: dict) -> str | None:
    rng = BLOOD_RANGES.get(item["feature"])
    if rng is None or item.get("value") is None:
        return None
    low, high, low_phrase, high_phrase = rng
    if item["value"] < low:
        return low_phrase
    if item["value"] > high:
        return high_phrase
    return None


def plan_queries(agent: dict) -> dict:
    """agent: the orchestrator's result (belief, tool_outputs, consistency)."""
    belief = agent["belief"]
    pred = belief["predicted_class"]
    out = {"predicted_class": pred, "queries": [], "skipped": [], "note": None}
    if pred not in CLASS_PHRASE:
        out["note"] = "Normal prediction: no abnormal drivers to explain, so no literature is requested."
        return out
    cls = CLASS_PHRASE[pred]

    def usable(modality: str):
        """A modality's attributions explain ITS OWN predicted class. If it disagrees with the final
        prediction, its drivers support a different class and must not seed queries for this one."""
        t = agent["tool_outputs"].get(modality)
        if not t:
            return None
        if t["predicted_class"] != pred:
            out["skipped"].append(f"{modality} model predicted {t['predicted_class']}, not {pred}: its drivers "
                                  f"explain a different class, so no {pred} query is built from them")
            return None
        return t["attributions"]["items"]

    # differential query first: when modalities disagree it is the most useful one, and the cap must not cut it
    # The rival is the disease class a disagreeing modality predicted (strongest first), because the
    # pooled runner-up can be Normal, which has no literature to ask for. Fall back to the runner-up.
    probs = belief["probabilities"]
    if agent["consistency"]["status"] in ("partial", "conflict"):
        rivals = sorted(((t["probabilities"][t["predicted_class"]], m, t["predicted_class"])
                         for m, t in agent["tool_outputs"].items()
                         if t["predicted_class"] != pred and t["predicted_class"] in CLASS_PHRASE), reverse=True)
        if rivals:
            _, m, rival = rivals[0]
            why = f"{m} model predicted {rival}"
        else:
            rival = max((c for c in probs if c != pred), key=probs.get)
            why = f"runner-up {rival} ({probs[rival]:.0%})"
        if rival in CLASS_PHRASE:
            out["queries"].append({"text": f"{cls} versus {CLASS_PHRASE[rival]} differential diagnosis imaging",
                                   "modality": "agent", "driver": f"modalities disagree; {why}", "effect": None,
                                   "terms": None})

    # lab drivers, strongest push first
    blood = usable("blood") or []
    blood_queries = []
    for it in sorted((i for i in blood if i["direction"] == "supports"), key=lambda i: -i["effect"]):
        phrase = _blood_phrase(it)
        if phrase:
            blood_queries.append({"text": f"{cls} with {phrase}", "modality": "blood",
                                  "driver": f"{it['feature']} = {it['value']}", "effect": it["effect"],
                                  "terms": TERMS.get(phrase)})
        else:
            out["skipped"].append(f"{it['feature']} = {it['value']}: supported the prediction but is within its "
                                  "reference range (or has no usable phrase), so no literature query")
    out["queries"] += blood_queries[:2]

    # image drivers: the two strongest clearly-present channels, one query each. A combined query
    # ("edema and lung opacity") scored 3.3 on the reranker where "lung opacity" alone scored 10.8,
    # because a cross-encoder rewards a passage that matches the whole query, and few passages do.
    img = usable("image") or []
    chans = [i for i in sorted(img, key=lambda i: -i["effect"])
             if i["direction"] == "supports" and (i.get("value") or 0) >= IMAGE_MIN_VALUE and i["feature"] in IMAGE_PHRASES][:2]
    for c in chans:
        finding = IMAGE_PHRASES[c["feature"]]
        out["queries"].append({"text": f"{cls} chest X-ray with {finding}", "modality": "image",
                               "driver": f"{c['feature']} = {c['value']}", "effect": c["effect"],
                               "terms": TERMS.get(finding),
                               # used only when the primary query finds nothing. A single passage about BOTH a
                               # finding and a disease often does not exist ("pneumonia chest X-ray with pulmonary
                               # edema" scored 1.1) while one that relates the two does ("pulmonary edema versus
                               # pneumonia chest radiograph" scored 5.1).
                               "fallback": f"{finding} versus {cls} chest radiograph"})

    # text drivers: clinically meaningful words that supported the prediction
    txt = usable("text") or []
    words = []
    for it in sorted((i for i in txt if i["direction"] == "supports" and i["effect"] >= 0.05), key=lambda i: -i["effect"]):
        w = "".join(ch for ch in it["feature"].lower() if ch.isalnum())
        if w in CLINICAL_TERMS and w not in SELF_WORDS[pred] and w not in words:
            words.append(w)
    if words:
        out["queries"].append({"text": f"{cls} with {' and '.join(words[:2])}", "modality": "text",
                               "driver": "indication words: " + ", ".join(words[:2]), "effect": None,
                               "terms": tuple(words[:2])})

    seen, uniq = set(), []
    for q in out["queries"]:
        if q["text"] not in seen:
            seen.add(q["text"]); uniq.append(q)
    out["queries"] = uniq[:MAX_QUERIES]
    return out
