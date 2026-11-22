# Results

Newest first. Every number here can be rebuilt with `atlas-eval` from the raw transcripts. Those live outside git, in `runs/`, until a round is frozen.

## M0, 22 Nov 2026: three baselines

**Setup.** Ten Hindi districts (`data/m0_districts.tsv`), 1,511 clips from 793 speakers (`data/manifests/m0.jsonl`), run on a Kaggle T4. Models: Whisper-large-v3 (faster-whisper), Omnilingual ASR `omniASR_LLM_3B_v2` with a Hindi hint, and IndicConformer 600M. WER uses `normalise()`, and lenient WER uses `lenient()`. Intervals are 95%, from 1,000 speaker resamples. Rows are ordered by the best model's WER, worst first.

| District | Whisper-large-v3 | Omnilingual 3B | IndicConformer |
|---|---|---|---|
| Kabirdham | 31.9% (29.4–34.5) | 23.4% (21.5–25.3) | 16.1% (14.7–17.8) |
| Rohtak | 29.7% (25.9–33.0) | 21.6% (18.1–25.0) | 16.1% (13.2–18.8) |
| West Champaran | 28.2% (23.9–33.3) | 20.5% (17.6–23.9) | 14.5% (12.2–17.1) |
| Dhar | 22.6% (19.0–25.8) | 15.0% (12.0–18.0) | 12.9% (10.2–15.4) |
| Patna | 23.7% (20.7–27.0) | 19.4% (15.8–23.3) | 12.6% (10.5–14.9) |
| Lucknow | 24.0% (21.6–26.5) | 16.0% (13.6–18.4) | 12.3% (10.3–14.5) |
| Garhwa | 22.2% (20.2–24.1) | 15.6% (13.7–17.7) | 11.8% (10.2–13.4) |
| Bikaner | 26.1% (23.9–28.5) | 18.4% (16.4–20.6) | 11.5% (10.0–13.2) |
| Katni | 20.1% (16.0–24.3) | 14.0% (10.6–17.1) | 11.5% (8.6–14.0) |
| Lalitpur | 21.7% (18.8–24.7) | 14.8% (12.5–17.2) | 10.2% (8.5–12.0) |
| **All ten** | **25.9%** | **18.6%** | **13.3%** |

Lenient WER on the same clips:

| District | Whisper-large-v3 | Omnilingual 3B | IndicConformer |
|---|---|---|---|
| Kabirdham | 25.9% | 16.9% | 11.4% |
| Rohtak | 26.2% | 18.7% | 12.3% |
| West Champaran | 24.7% | 16.9% | 10.9% |
| Dhar | 19.4% | 11.6% | 8.9% |
| Patna | 20.3% | 16.0% | 8.6% |
| Lucknow | 20.4% | 13.5% | 8.8% |
| Garhwa | 18.6% | 11.8% | 7.5% |
| Bikaner | 22.2% | 14.9% | 7.9% |
| Katni | 16.7% | 10.5% | 7.4% |
| Lalitpur | 18.3% | 11.4% | 7.0% |
| **All ten** | **22.0%** | **14.7%** | **9.3%** |

**Gate.** Passes for all three models, under both measures:

| Model | Measure | Worst ÷ best | Intervals |
|---|---|---|---|
| Whisper | WER | 1.58× (Kabirdham vs Katni) | Don't overlap |
| Whisper | Lenient | 1.57× (Rohtak vs Katni) | Don't overlap |
| Omnilingual | WER | 1.67× (Kabirdham vs Katni) | Don't overlap |
| Omnilingual | Lenient | 1.78× (Rohtak vs Katni) | Don't overlap |
| IndicConformer | WER | 1.58× (Kabirdham vs Lalitpur) | Don't overlap |
| IndicConformer | Lenient | 1.77× (Rohtak vs Lalitpur) | Don't overlap |

The label audit still has to confirm it.

**Findings:**

- **All three models find the same districts hard.** Kabirdham, Rohtak and West Champaran are the worst three for every model. Rank correlations between models are 0.95 (Whisper and Omnilingual), 0.83 (Omnilingual and IndicConformer) and 0.82 (Whisper and IndicConformer). So the effect belongs to the districts, not to one model. It could still be the districts' *transcribers* rather than their speech, though; the audit separates the two.
- **IndicConformer wins every district, and every lead is real.** A paired speaker bootstrap puts Omnilingual behind it by 2.1 points in Dhar (0.2 to 4.3) up to 7.2 points in Kabirdham (5.3 to 9.0). E2 ("no model wins more than 80% of districts") fails on M0. The catch: IndicConformer's model card doesn't say what it was trained on. If that includes Vaani, its lead is inflated, so settle contamination before M1 freezes the baselines.
- **Kabirdham's transcripts need checking first.** In 93 of its 151 clips, all three models agree on a word the reference doesn't have; elsewhere it's 20 to 41 clips per district. Many are words all three models drop (है, एक, अ), and spellings such as बहौत and गयी. So part of Kabirdham's top rank may come from its transcripts, not its speech. The listening audit samples 15 of these clips per district (`audit/`).
- **Omnilingual writes formal Hindi.** Its top substitutions turn Vaani's spoken forms into written ones: बहोत→बहुत (148), पे→पर (108), ये→यह (94), plus है/हैं both ways (355). Lenient scoring forgives these and takes it from 18.6% to 14.7%. It wrote no Latin script and produced no runaway output.
- **About a third of IndicConformer's errors are spelling, not hearing.** The main ones are है/हैं (401), Vaani writing the spoken बहोत (148), and dropped nasal marks such as मे/में. Lenient scoring removes them, and the district spread holds.
- **Whisper's errors are mostly near-miss spellings.** 78% are substitutions: aspiration and vowel-length slips such as छोटे→चोटे, सामान→समान, ऊपर→उपर and दिखाई→देखाई, on top of the same variants. 99% of its output is Devanagari, and there were no runaway hallucinations.
- **The best open model is already at 10–16% WER.** Absolute-point targets for Kaan won't work, so convert them to relative ones at the M0 revision (see EXPECTED_RESULTS.md).
- **Dialect replacement (E3) has too little data here.** Vaani glossed only 91 dialect words with their standard Hindi in this slice. E3 needs the planned hand-labelled errors.

**Still open for M0:** the listening audit (300 clips picked in `data/manifests/m0_audit.jsonl`; needs a Hindi-speaking listener), 50 errors per model read for the taxonomy, and IndicConformer's training data.
