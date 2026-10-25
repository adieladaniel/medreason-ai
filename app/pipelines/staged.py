"""
The case pipeline as separate stages, so the answer can arrive before the explanation does.

  predicting   each modality's class probabilities, then the belief, the agreement check and the
               missing-information plan. No attribution step, so this is the part a user waits for.
  explaining   SHAP and Grad-CAM per modality, quickest first, saved one modality at a time. The text
               SHAP run (under a second typically, 10 to 17 s for a 100-word indication) is why it is a separate stage.
  retrieving   evidence planner, then retrieval of numbered passages
  writing      the local model writes one sentence per passage (case sentences are built by code)
  verifying    NLI, number and wording checks; only verified sentences are kept; final text composed

Why split: run_case computes the prediction and its explanation together per modality, so nothing is
shown until the slowest explainer finishes. Here only `predicting` is on the critical path. The belief
comes from the same function as run_case (app/agent/reasoning.py) fed the same models' probabilities, and
the explaining stage re-derives each modality's probabilities and records whether they match.

Design rule kept: the prediction is fixed after `predicting`. Later stages read it and add text around
it; none of them writes a class. If an explainer's probabilities ever differ from the stored prediction,
the difference is recorded, and the stored prediction stays what the user sees.

A stage is a function of one dict (the context) that returns JSON-ready data. The context holds the case
inputs, the results of earlier stages, and `deps` (the model callables). Tests pass fakes in `deps`, so
none of this needs a model to be tested; with no deps given, the real models are used.
"""
from __future__ import annotations

import re
import time
from collections import namedtuple
from pathlib import Path

from app.agent import MODALITY_PRIORITY
from app.agent.decision import decide
from app.agent.reasoning import reason
from app.config import get_settings
from app.rag.evidence import assemble
from app.rag.explain_case import compose
from app.rag.faithfulness import verify
from app.rag.planner import plan_queries
from app.rag.writer import write_explanation

Stage = namedtuple("Stage", "name field fn")

# quickest first, so blood and image explanations are on screen while the text SHAP run is still going
EXPLAIN_ORDER = ["blood", "image", "text"]
# explanation probabilities and stored prediction probabilities are both rounded to 4 places
PROB_TOLERANCE = 1e-4


def provided_inputs(indication: str | None, labs: dict | None, image_path: str | None) -> dict:
    """The modalities a case actually has. A blank indication, one with no letters or digits ("???"), or labs
    with no value in them count as absent: someone who left the field empty gave no information, and
    feeding the model nothing would add a confident-looking opinion that is built on nothing (an empty
    indication scores Pneumonia and "???" scores Lung Cancer)."""
    out = {}
    if image_path:
        out["image"] = image_path
    if labs and any(v is not None for v in labs.values()):
        out["blood"] = labs
    if indication and re.search(r"\w", indication):
        out["text"] = indication
    return out


def _tools():
    from app.agent import tools          # imported here: it loads the model libraries
    return tools


# ------------------------------------------------------------------ stage 1
def stage_predict(ctx: dict) -> dict:
    predictors = ctx["deps"].get("predictors") or _tools().PREDICTORS
    preds, seconds = {}, {}
    for m in MODALITY_PRIORITY:                      # text first: the cheapest, so a failure shows early
        if m in ctx["inputs"]:
            t0 = time.perf_counter()
            preds[m] = predictors[m](ctx["inputs"][m])
            seconds[m] = round(time.perf_counter() - t0, 3)
    reasoned = reason({m: p["probabilities"] for m, p in preds.items()})
    cfg = get_settings()
    # how sure the system is; read from the pooled probabilities and never changes the predicted class
    decision = decide(reasoned["belief"]["probabilities"], list(preds), set_coverage=cfg.prediction_set_coverage,
                      alert_recall=cfg.cancer_alert_recall)
    return {**reasoned, "decision": decision,
            "modalities": {m: {"predicted_class": p["predicted_class"], "probabilities": p["probabilities"]}
                           for m, p in preds.items()},
            "modalities_used": [m for m in MODALITY_PRIORITY if m in preds], "seconds": seconds}


# ------------------------------------------------------------------ stage 2
def _save_figure(plotter, res: dict, case_id: str, modality: str, figure_dir: Path | None) -> dict:
    """Best effort: a figure that fails to draw must not take the explanation down with it."""
    if plotter is None or figure_dir is None:
        return {"figure": None}
    try:
        figure_dir.mkdir(parents=True, exist_ok=True)
        plotter(res, figure_dir / f"{modality}.png")
        return {"figure": f"{case_id}/{modality}.png"}
    except Exception as exc:                          # noqa: BLE001
        return {"figure": None, "figure_error": f"{type(exc).__name__}: {exc}"}


def stage_explain(ctx: dict) -> dict:
    explainers = ctx["deps"].get("explainers") or _tools().TOOLS
    plotters = ctx["deps"].get("plotters") or {}
    stored = ctx["prediction"]["modalities"]
    out: dict = {}
    for m in EXPLAIN_ORDER:
        if m not in ctx["inputs"]:
            continue
        t0 = time.perf_counter()
        res = explainers[m](ctx["inputs"][m])
        diff = max(abs(res["probabilities"][c] - stored[m]["probabilities"][c]) for c in stored[m]["probabilities"])
        entry = {k: v for k, v in res.items() if not k.startswith("_")}      # drop heatmaps and raw arrays
        entry.update(seconds=round(time.perf_counter() - t0, 3), matches_prediction=diff <= PROB_TOLERANCE,
                     max_probability_difference=round(diff, 6))
        entry.update(_save_figure(plotters.get(m), res, ctx["case_id"], m, ctx.get("figure_dir")))
        out[m] = entry
        ctx["progress"](out)                          # blood and image show up while text SHAP still runs
    return out


# ------------------------------------------------------------------ stages 3 to 5
def build_agent(ctx: dict) -> dict:
    """The shape the evidence chain reads: the stored belief plus the explainers' attributions."""
    p = ctx["prediction"]
    return {"belief": p["belief"], "consistency": p["consistency"], "missing_info_plan": p["missing_info_plan"],
            "tool_outputs": ctx["explanations"]}


def stage_retrieve(ctx: dict) -> dict:
    plan = plan_queries(build_agent(ctx))
    ev = assemble(plan)
    return {"plan": plan, "passages": ev["passages"], "unanswered": ev["unanswered"]}


def _passages(ctx: dict) -> dict:
    return {"passages": ctx["evidence"]["passages"], "unanswered": ctx["evidence"]["unanswered"]}


def stage_write(ctx: dict) -> dict:
    kw = {"recorder": ctx.get("recorder")}
    if ctx["deps"].get("model"):
        kw["model"] = ctx["deps"]["model"]
    written = write_explanation(build_agent(ctx), ctx["evidence"]["plan"], _passages(ctx), **kw)
    return {**ctx["evidence"], "written": written}


def stage_verify(ctx: dict) -> dict:
    result = verify(ctx["evidence"]["written"], _passages(ctx))
    final = compose(ctx["evidence"]["plan"], _passages(ctx), result)
    final.pop("plan", None)                          # already stored once, under evidence.plan
    return {**ctx["evidence"], "final": final}


def default_stages() -> list[Stage]:
    """The first stage is the one that must work: without a prediction the case has no answer. A failure
    in any later stage leaves the case with what exists so far (status "partial")."""
    return [Stage("predicting", "prediction", stage_predict),
            Stage("explaining", "explanations", stage_explain),
            Stage("retrieving", "evidence", stage_retrieve),
            Stage("writing", "evidence", stage_write),
            Stage("verifying", "evidence", stage_verify)]


def default_plotters() -> dict:
    """Figure functions of the three explainers (imported here because they load matplotlib and SHAP)."""
    from explain import blood, image, text
    return {"blood": blood.plot_blood, "image": image.plot_image, "text": text.plot_text}


def warm_up() -> dict:
    """Load every model once, so the first real case does not pay for it (the first run after a start
    took about 90 seconds). Returns seconds per model. Memory, measured: 0.5 GB for Python with torch, 1.2 GB once every
    model here is loaded, 2.3 GB after one full case; the language model in Ollama is separate (1.2 GB of RAM and 2.4 GB of
    the 4 GB GPU)."""
    import explain.blood as b, explain.image as i, explain.text as t
    from app.rag import faithfulness, hybrid, retrieve
    steps = {"image model": i._load, "blood model": b._load, "text model": t._load,
             "evidence index": retrieve.retriever, "reranker": lambda: hybrid._reranker(retrieve.RERANKER),
             "entailment checker": faithfulness._model}
    timings = {}
    for name, load in steps.items():
        t0 = time.perf_counter()
        load()
        timings[name] = round(time.perf_counter() - t0, 1)
    return timings
