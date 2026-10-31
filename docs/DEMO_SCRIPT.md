# Demo script and recording guide

This file is for whoever presents the project, and it assumes you run the recording yourself. It has the rule about what may appear on screen,
the preparation, a script of about 80 seconds for a public video, and a longer tour for showing the project in person. Every number and every
phrase quoted from the page was observed on 2026-10-30 with the real models and Ollama (llama3.2), on the development laptop (RTX 3050 with 4 GB, 16 GB of RAM, Windows 11).

## 1. What may appear on screen

The nine validation demo cases show real MIMIC-CXR films and MIMIC clinical text. The PhysioNet Credentialed Health Data License (version 1.5.0,
the text shown on the MIMIC-CXR and MIMIC-IV license pages) says the licensee "will not share access to PhysioNet restricted data with anyone
else" (term 3) and "will use the data for the sole purpose of lawful use in scientific research and no other" (term 6). It does not say whether a
recording that shows the data is allowed. Nobody here has asked PhysioNet, so treat the answer as "not allowed" until they say otherwise.

The rule that follows:

- A public video, a screenshot or a post shows no MIMIC film and no MIMIC text. Not in a corner, not for a second.
- The public video uses a public-domain chest film and inputs typed for the occasion (section 3), and says so.
- Pages built from aggregate numbers are fine: Overview, Data, Models, Fusion, Reliability, Evidence, Roadmap. They hold no study, patient or admission
  identifiers.
- The Explainability page is not fine. It is a gallery of real validation cases with their films and indication text. Never open it on camera, and
  hide the folder it reads while recording (section 2).
- The Recent cases list on the Analyze page is fine only if the database holds nothing but the cases you ran for the recording (section 2).
- Showing the nine cases in person (a screen share in an interview, for example) is a grey area for the same reason. If it matters, ask PhysioNet
  support first. Section 5 is written for that use and is for private use only.

## 2. Preparation, about 15 minutes

Close Chrome, WhatsApp and anything else heavy first. Docker's WSL machine can hold up to 8 GB of the 16 GB, so stop Docker Desktop too. On this
laptop a case takes a few seconds when it is quiet and several times longer when memory is tight (see the latency notes in experiments/e2e/RESULTS.md).

1. Get the public film, once. It needs the network:

   ```powershell
   python scripts/prepare_public_demo.py
   ```

   This writes `data/demo_public/chest_pa_cdc17298.png` and `CREDIT.txt`. The film is a posteroanterior chest radiograph from the CDC Public Health
   Image Library, image 17298, credit "CDC/ Dr. Jack Poland", listed as public domain with credit requested. The script converts it to one grayscale
   channel because the JPEG carries a slight colour tint and the app's non-X-ray guard refuses colour pictures. Put the credit line in the video or
   its description.

2. Hide the sample gallery, and restore it afterwards:

   ```powershell
   Rename-Item experiments\explain_samples explain_samples_hidden
   # after recording:
   Rename-Item experiments\explain_samples_hidden explain_samples
   ```

3. Start Ollama and check it answers. Running `ollama list` starts the Ollama app, which starts the server. When you finish, stop all three
   processes (`Get-Process ollama, "ollama app", llama-server | Stop-Process`) and check `nvidia-smi` shows about 100 MiB used.

4. Start the server on its own empty database, with no demo cases offered, so nothing from MIMIC can be loaded by a stray click:

   ```powershell
   $env:MEDREASON_DATABASE_URL = "sqlite:///data/demo_recording.db"
   $env:MEDREASON_UPLOAD_DIR = "data/demo_recording_uploads"
   $env:MEDREASON_DEMO_DIR = "data/no_demo_cases"
   $env:MEDREASON_WARM_UP_MODELS = "true"
   $env:MEDREASON_REBUILD_DASHBOARD = "false"
   python -m uvicorn app.main:app --port 8000 --no-access-log
   ```

   Open http://127.0.0.1:8000, not `localhost` (on this laptop `localhost` costs about 2 seconds per call). `/api/ready` should say `ok`. The
   example menu then lists only the two built-in non-X-ray pictures.

5. Run the two cases below once before you record, with no camera. The first case after a start is slow while the models load (5.5 s to the
   prediction and 12.5 s to the end in the test run), and a case with a literature note took 57 s the first time. Leave both finished cases in the
   database: the recording opens them from the Recent cases list.

   - Routine case. Film: `data/demo_public/chest_pa_cdc17298.png`. Indication: `Routine pre-operative chest film. No cough, no fever.`
     Labs: WBC 7.2, hemoglobin 14.1, platelets 250, albumin 4.2, sodium 140, potassium 4.1, glucose 92, creatinine 0.9.
   - Pneumonia-like case. Same film. Indication: `Cough, fever and shortness of breath. Concern for pneumonia.`
     Labs: WBC 24.0, neutrophils 90, lymphocytes 5, hemoglobin 11.0, albumin 2.5, LDH 380, CRP 180, sodium 133, glucose 130, creatinine 1.2.

Browser at 100 percent zoom, a window about 1366 by 768, bookmarks bar hidden, notifications off.

What the two cases showed in the test run, so you can tell if something is off. The routine case: Normal at 77 percent, "All three agree", the
prediction on screen after 0.9 s and the whole case finished after 1.9 s. The pneumonia-like case: Pneumonia at 63 percent, "Two agree. Image is
the outlier, predicting Normal", prediction after 0.9 s, finished after 57 s, with one cited sentence and "2 drafted sentence(s) were removed
because they did not match the passage they cited". The written note is not reproducible, so the sentence and the count can differ on your run.
If your answer differs, narrate what is on the screen, not what this script says.

How these typed inputs were chosen, because it matters: the first pneumonia-like phrasing I tried ("Fever and productive cough for four days.
Evaluate for pneumonia.", eight labs) gave Normal at 56 percent with the blood model dissenting. I did not hide that. I tried two stronger
phrasings and used the first of them. So the pneumonia-like case is an example chosen to show the evidence note, not evidence that the system
recognises pneumonia. Do not present it as either.

## 3. The 80-second public video

Narration is about 190 words, which is 80 to 90 seconds at an unhurried pace (140 words a minute) with short pauses while the page changes.
Say it in your own words, but keep the facts.

| Time | On screen | Say |
| --- | --- | --- |
| 0:00 to 0:12 | The Analyze page, the blue disclaimer banner in view. | "This is MedReason-AI, a student project, not a medical device. It takes a chest X-ray, the reason for the exam and blood labs, and predicts Normal, Pneumonia or Lung Cancer." |
| 0:12 to 0:26 | Recent cases, open the routine case, click Edit and re-run, then Analyze. The stage strip fills. | "These are typed examples on a public-domain film, not a patient. The prediction appears in about a second: Normal, 77 percent, all three inputs agree. Explanations follow." |
| 0:26 to 0:56 | Recent cases, open the pneumonia-like case. Show the prediction card and the line "Two agree. Image is the outlier". Scroll to the three input cards, then to the literature section. | "Now a case where the inputs disagree. Labs and indication point to pneumonia, the film looks normal to the image model. The answer is Pneumonia, 63 percent, and the page says the image is the outlier, and that cases like this were right about two times in three on validation. Each input shows what it used. The literature note is checked against its source, and it is not proof about this patient." |
| 0:56 to 1:07 | New case. Open the example menu, choose "Not an X-ray: a colour picture", click Analyze. The refusal appears. | "Something that is not an X-ray is refused before any model runs." |
| 1:07 to 1:27 | The Reliability page, then the footer line "The sealed test split was scored once: macro F1 0.609". | "The numbers are measured. On validation the combined score is 0.61. The test set was scored once and gave 0.609. The text input carries most of the signal and the image adds almost nothing, and the write-up says so. FastAPI, Postgres, Qdrant, Docker and CI sit underneath." |

The refusal text on the page is: "This looks like a colour photograph or screenshot, and chest X-rays are grayscale. Send a chest X-ray as a PNG or JPEG."
The test run's browser also logs one 422 console error at that moment. That is the refusal reaching the page, not a fault.

Cut points. The literature note takes up to a minute when made live, so the script opens a finished case. Say so on camera, as above ("typed
examples"), and never show a made-up wait as if it were real. If you want one live run with a note, record it, speed that stretch up and say so.

## 4. What to claim, and what not to

You can say, and the repository backs each one:

- Test macro F1 0.609 (interval 0.559 to 0.654) on 2,099 studies from 721 patients, scored once. Validation was 0.613.
- Calibration error 0.118 raw and 0.029 after temperature scaling on the test split, and the prediction sets held their coverage (81.4, 79.4 and 80.8
  percent against a target of 80).
- The engineering: a staged pipeline that shows the prediction first, retrieval with a verifier that removes unsupported sentences, a locked and hashed
  dependency set, database migrations, and a CI job that starts the real containers and tests them.
- The system says "uncertain" and shows what it cannot rule out, and it reports when it has no literature.

Do not say:

- "It diagnoses" or "it reads X-rays". The text input carries most of the signal, a keyword rule scores about the same as the text model, and on the
  test split the image's contribution is indistinguishable from zero (paired gain +0.005, interval -0.013 to +0.021).
- "61 percent accurate". Macro F1 is not accuracy. Accuracy on the test split was 0.704 against 0.628 for always guessing the majority class.
- That it works on other conditions. It cannot say "none of these": on other pathologies it answered without a low-confidence flag 72.4 percent of
  the time.
- That the typed pneumonia-like example shows it recognises pneumonia (section 2).
- Anything clinical. The project is a foundation for later work, not a product.

## 5. The longer tour, for private use

Needs the `demo/cases` folder (git-ignored, MIMIC data) and the section 1 caveat. Before presenting, run `python scripts/rehearse_demo.py --url
http://127.0.0.1:8000 --keep` against the running stack. It sends all nine cases the way the page does and compares each with its frozen values, and
`--keep` leaves them in the Recent cases list so you can open them instantly. Times are from the first real run with the models still loading.

| Order | Case | What it shows | Seconds |
| --- | --- | --- | --- |
| 1 | clear-normal | An easy Normal: 91 percent, all three agree, the prediction set holds one class. | 48 (first case) |
| 2 | clear-pneumonia | An easy Pneumonia: 92 percent, all agree, with one cited literature sentence and two dropped by the checker. | 100 (first note) |
| 3 | clear-lung-cancer | Lung Cancer found at 84 percent, two of three agree, the cancer alert fires. The note says the library has no passage for this finding: the known corpus gap, reported and not hidden. | 9 |
| 4 | models-disagree-resolved | The consistency checker reports a conflict, and the pooled answer, Normal at 71 percent, is right. | 4 |
| 5 | low-confidence | Normal at 47 percent, headlined as uncertain, the set holds Normal and Pneumonia. | 4 |
| 6 | no-labs | Image and text only: the blood model takes no part. Normal at 93 percent. | 11 |
| 7 | wrong-but-flagged | True Lung Cancer, answered Normal at 35 percent, but flagged as uncertain and the cancer alert fires. | 7 |
| 8 | missed-cancer-alert-fires | True Lung Cancer, headline Pneumonia at 46 percent, the set includes Lung Cancer and the alert fires. | 17 |
| 9 | wrong-and-confident | True Pneumonia, answered Normal at 92 percent with all inputs agreeing and no warning. The honest failure: the safeguards do not catch everything. | 14 |

End on case 9 if you are asked about limits. It is the strongest evidence that the numbers were not tuned to look good.

## 6. Recording

Windows 11 22H2 should have a screen recorder in the Snipping Tool (Win+Shift+R) and the Xbox Game Bar (Win+Alt+R); neither was tried here, so check which
your machine has. OBS Studio is free and reliable if neither works. Record the browser window only, at 1920 by 1080 if the screen allows, with a
separate voice track so you can redo the narration without redoing the screen. Keep the cursor slow, pause a beat after each page changes, and trim the
start and end. Check the finished file frame by frame for any MIMIC film or text before it leaves your machine.
