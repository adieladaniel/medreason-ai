# What MedReason-AI achieves, and what it does not

Version 1.5, as of 2026-10-29. This page is the honest summary: the headline numbers on patients the system had never seen, what each
part contributes, where it fails, and what was tried and did not work. Every figure links to the experiment that produced it. The system is
a demonstration of how to build a multimodal medical AI service, not a clinical tool.

## The result

Given a chest X-ray, the blood labs on record and the clinical indication for the exam, the system names Normal, Pneumonia or Lung Cancer.
Scored once on a sealed test split of **2,099 studies from 721 patients who appear nowhere in training or validation**
([details](../experiments/test_evaluation/RESULTS.md)):

| | Sealed test | Validation |
|---|---|---|
| macro F1 (the number to quote) | **0.609** (0.559 to 0.654) | 0.613 |
| F1 Normal / Pneumonia / Lung Cancer | 0.629 / 0.775 / 0.421 | 0.615 / 0.797 / 0.426 |
| accuracy (always answering Pneumonia would score 0.628) | 0.704 | 0.719 |
| Lung Cancer ROC AUC | 0.841 (0.767 to 0.904) | 0.853 |
| calibration error, raw / after temperature scaling | 0.118 / 0.029 | 0.142 / 0.026 |

Intervals are 95 percent, resampling patients. The validation numbers were good estimates: the drop is 0.004. In plain words, this
system is right about 7 times in 10 and, because the classes are unbalanced, clearly better than guessing but far from a reliable
diagnostic, with Lung Cancer found imperfectly (F1 0.42, 120 cases).

## What each input contributes

| Adding this input to the other two | Validation | Sealed test |
|---|---|---|
| text (the indication) | +0.105 (+0.063 to +0.143) | +0.125 (+0.083 to +0.163) |
| blood labs | +0.051 (+0.023 to +0.078) | +0.029 (-0.001 to +0.057) |
| image | +0.026 (+0.011 to +0.042) | +0.005 (-0.013 to +0.021) |
| all three over text alone | +0.104 (+0.074 to +0.133) | +0.049 (+0.014 to +0.081) |

Differences in macro F1, paired on the same resampled patients. Three things follow, and they matter more than the headline:

1. **The text input carries the result, and it is mostly the referring clinician's suspicion in words.** A crude keyword rule scores
   about the same as the trained text model (0.502 against 0.509 on validation). It is a legitimate input, known before the film is read,
   but it is clinical context, not the AI understanding a report.
2. **Combining inputs helps, by about half as much as validation suggested.** Fusion still beats the best single input on unseen patients.
3. **The image contributes little.** It cannot see Lung Cancer (ROC AUC 0.53 on validation, 0.60 on test with a wide interval), and on test
   its contribution to the fused result cannot be told from zero. Swapping the frozen image backbone for ones with trained Nodule and Mass
   outputs did not fix this ([experiment](../experiments/image_backbones/RESULTS.md); nothing was adopted).

## The safeguards, on unseen patients

| | Result on the sealed test split |
|---|---|
| Declining the least confident cases (low-confidence flag) | answers 81.9 percent of cases; accuracy on those rises from 70.4 to 75.2 percent |
| Prediction sets ("it could be one of these") at 80 percent | true class inside the set for 81.4 percent of Normal, 79.4 percent of Pneumonia, 80.8 percent of Lung Cancer studies |
| Cancer alert, default setting | flags 56.7 percent of true cancers (44.0 to 69.0), 11 alerts per 100 cases, 29 percent of alerts real |
| Consistency check, "all three agree" / "two agree" / "all three differ" | 85.5 / 64.2 / 53.7 percent correct |

These were measured on validation and held on data nobody had touched, which is the best evidence in the project that the measurement
discipline worked. The alert trades false alarms for catches (four settings are reported); it states the model's signal and is not a diagnosis.

## Where it fails

- **It cannot say "none of these".** The three classes are closed: 10,049 studies with some other chest condition were dropped when the dataset
  was built. On 1,500 such studies from patients the system had never seen, 72 percent were answered without a low-confidence flag, 52 percent
  were named Pneumonia, 38 percent Normal, and confidence told them apart from in-scope studies only weakly (area under the ROC curve 0.62).
  The cancer alert fired on 18 percent of them, against 10 percent of in-scope non-cancer studies
  ([measurement](../experiments/other_pathology/RESULTS.md)). A gate trained to flag such studies from the system's own outputs was then
  tried and did not work (held-out AUROC 0.57, flagging 27 percent of ordinary studies; its cross-validated 0.74 came from a mismatch in how
  the training rows were built, and on clean rows it was 0.60; [experiment](../experiments/out_of_scope/RESULTS.md)). The information is not in
  those outputs, so a real fix means teaching the models an "other" outcome, which is future work.
- **The labels are noisy.** They come from admission-level billing codes and report mentions, so a film may show nothing of the diagnosis it
  is labelled with. Scores measure agreement with those labels.
- **One hospital system.** Same linking rule and labeller for training and test. This is not external validation.
- **The image backbone may have seen these patients in pretraining** (an open caveat; the experiment that tried to clear it could not).
- **The evidence layer is the least validated part.** The retrieval was tested on 54 queries written by the author with the assistant, the
  written note is produced by a 3 billion parameter local model whose output varies between runs, and the sentence checker was validated on a
  small labelled set with only three real unfaithful sentences. About one faithful sentence in five is dropped, and a sentence that blends true
  facts into a false statement can pass. The page therefore presents background as what a cited paper says, never as proof about the patient.
- **Uploads are only roughly screened.** A guard refuses what is clearly not a radiograph (colour photographs, blank or flat images,
  gradients, pages of text, noise: all refused in tests, 0.35 percent of real validation films wrongly refused). It does not check that a
  film is a chest or frontal view, and it passes about half of grayscale photographs ([measurement](../experiments/xray_guard/results.json)).
- **No login, no TLS, one worker.** A local demonstration with no real patient data.

## Speed

On a laptop in ordinary use, the prediction is on screen in about half a second. A full note with verified citations takes about 15 to 20
seconds (up to about 50 for the first case after a start); a Normal answer needs no note and takes 1.5 to 2 seconds. The slowest step is the
sentence checker on the CPU (about 3.4 seconds per drafted sentence) ([run](../experiments/e2e/RESULTS.md)).

## What was tried that did not work

- Feature-level fusion (one network over all the inputs) scored lower than averaging the three models' probabilities (0.586 against 0.613).
- A better image backbone for cancer (NIH and PadChest weights): no gain that met the rule fixed in advance.
- Query expansion for retrieval, parent-child passage expansion for the writer, a second writer model, and two alternative sentence checkers:
  none improved on what was already in place.

## What would come next

1. Teach the models an "other" outcome and measure it on held-out other-pathology studies; this is the largest honest gap, and a gate on
   the existing outputs has been shown not to be enough.
2. Fine-tune or replace the image model with one trained on this task; the frozen-output approach has probably reached its limit.
3. Test at a second site, or on a public set such as NIH ChestX-ray14, for external validity.
4. Move the sentence checker to a GPU and enlarge the labelled set that validates it.

## How the numbers were kept honest

The test split was sealed until one final run; the plan, the frozen system and what would be reported were written down first
([plan](../experiments/test_evaluation/PLAN.md)); experiments were pre-registered with a fixed adoption rule
([image backbones](../experiments/image_backbones/PLAN.md), [other pathology](../experiments/other_pathology/PLAN.md)); every behaviour
change has a test that was shown to fail when the code was broken on purpose (over 400 tests; the last count is in
[CLAUDE.md](../CLAUDE.md)); and results that came out worse than hoped are reported next to the ones that came out well.
