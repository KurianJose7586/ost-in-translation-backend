# Results

Newest first. Every number here can be rebuilt with `atlas-eval` from the raw transcripts. Those live outside git, in `runs/`, until a round is frozen.

## M1, 23 Nov 2026: test sets from Vaani's official split (drafts)

**Setup.** `data/m1.py` reads every row of the test split in `ARTPARK-IISc/Vaani-transcription-part`, small columns only. 64 of its 90 languages have a test split; the other 26 have train only. The rules are M0's, with one change: a real speaker ID, under 40 s, a non-empty normalised transcript, no speaker in two districts, and at most 4 clips per speaker instead of 8 (see decisions.md). A district qualifies with 100 usable clips, and its draft takes 150 where it can. Counts for every language and district are in `data/m1_test_survey.tsv`. The drafts are `data/manifests/m1_{language}.jsonl`, not frozen yet.

| Language | Districts in test split | Qualifying | Clips | Speakers | Hours |
|---|---|---|---|---|---|
| Hindi | 130 | 93 | 13,877 | 3,936 | 24.7 |
| Kannada | 14 | 13 | 1,973 | 546 | 3.9 |
| Telugu | 23 | 13 | 1,963 | 537 | 3.5 |
| Bengali | 34 | 12 | 1,816 | 514 | 2.9 |
| Marathi | 24 | 8 | 1,212 | 344 | 2.1 |
| Chhattisgarhi | 9 | 5 | 731 | 206 | 1.3 |
| Bhojpuri | 14 | 5 | 706 | 216 | 1.4 |
| Odia | 5 | 4 | 606 | 152 | 1.0 |
| Maithili | 20 | 4 | 555 | 160 | 1.1 |
| 21 more languages, 1–3 districts each | 101 | 35 | 4,931 | 1,292 | 7.4 |
| **All 30** | **374** | **192** | **28,370** | **7,903** | **49.4** |

**Findings:**

- **The atlas can cover 149 of the test split's 162 districts** in at least one language. Hindi alone covers 93. Hindi was also recorded outside the Hindi belt, so its map includes districts such as Bangalore and Krishna.
- **A Vaani speaker is the name's speaker part, not the speaker and session together.** The split repository has no speakerID column. The main repository's ID is the state's and district's first two letters plus the speaker part of the audio file name (Patna: `BIPA_…`). This reproduces all 793 M0 speaker hashes, so one person has the same hash in M0 and M1. Taking the following session part as well would count 102,825 speakers in Hindi's test split instead of 11,481.
- **The official split keeps speakers apart, not images.** In Hindi, every generic test image and 73% of district-specific ones also appear in train. 80% of the Hindi draft's clips describe an image that train clips also describe. In the other large languages, 95–100% of generic-image clips and 20–79% of specific-image clips share their image with train. Dropping those clips would leave 67 Hindi districts with 100 clips, against 93. This is open (decisions.md).
- **Every qualifying district is on the map.** `data/vaani_districts.tsv` gained 16 districts that only the split repository has, mostly in Kerala, Tamil Nadu and Telangana.
- **22 M0 speakers are in the Hindi draft.** M0 drew from all three splits, so that is expected; M0 isn't training data.

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

**Error taxonomy.** A random 50 of each model's word errors (strict WER), read as text without the audio, so what was said is a judgement from context. Every one is in `notes/m0_errors.tsv`, with its clip.

| Kind | Whisper | Omnilingual | IndicConformer |
|---|---|---|---|
| Misheard: a different word, or garbled | 16 | 13 | 13 |
| Spelling variant that lenient WER forgives (है/हैं, बहोत/बहुत, ये/यह) | 6 | 14 | 8 |
| Other spelling of the same word (दीवार/दिवार, वगैरा/वगैरह) | 14 | 9 | 5 |
| Word boundary: joined or split words, acronyms letter by letter | 4 | 4 | 9 |
| Short word dropped or added (है, में, और) | 2 | 4 | 4 |
| Grammatical ending (रहे/रही, बच्चे/बच्ची) | 2 | 1 | 4 |
| Disfluency: a repeat, filler or cut-off word | 2 | 2 | 2 |
| Dialect form written as standard Hindi | 0 | 3 | 2 |
| Number in digits | 2 | 0 | 0 |
| English in Latin script | 1 | 0 | 0 |
| Reference probably wrong | 1 | 0 | 3 |

- **Only a quarter to a third of errors are plain mishearing.** The rest are mostly about writing: spelling accounts for 20, 23 and 13 of 50, word boundaries for 4, 4 and 9.
- **IndicConformer's boundary errors are mostly acronyms.** It writes ई सी ई where Vaani writes ईसीई, and डी ए एम सी for डीएमसी; the rest are compounds such as जो कि and जोकि. A lenient rule that joins runs of single letters would take most of these. It would change published numbers, so decide it together with the audit's spelling findings.
- **Whisper's spelling errors are near misses that lenient WER doesn't cover**: दीवार→दिवार three times in 50, plus vowel length, aspiration and doubled consonants.
- **Dialect forms are 5 of 150 errors**: री→रही, जे→जहां, पत्थल→पत्थर, ए→ये, and पीना for a spoken पिन्हा. That is far below E3's "more than a quarter". But text can't show dialect that Vaani's transcribers already wrote as standard Hindi. In one West Champaran clip the reference has पहना where the speaker seems to say पिन्हा, so E3 needs the audio.
- **The reference looks wrong in 4 of 150**: रावट for टावर, गोल for गोल्ड, कल for काले, and a phrase the reference seems to drop. The audit measures this properly.
- **Digits and Latin script are rare.** They make up 0.4% and 2.9% of all Whisper's errors, and none of the other two models'. The normaliser leaves both as errors.

**Contamination and Vaani's official split** (sources checked 22 Nov 2026):

- **No source lists Vaani in IndicConformer's training data, and none rules it out.**
  - The 600M model card says only that it covers 22 languages, and its metadata names no datasets. AI4Bharat's GitHub and AIKosh name none either.
  - Shaip, one of Vaani's three transcription vendors, also worked on IndicVoices.
  - On Vaani-Benchmark-V1.0, IndicConformer's published Hindi WER is 13.2, against 13.3% here. The two can't be compared directly: the benchmark scores against the closest of three references, which lowers WER, and its clips and normalisation differ.
- **Vaani has an official split, in a different repository from the one M0 used.** `ARTPARK-IISc/Vaani-transcription-part` has train, validation and test for Hindi (177, 52 and 52 files, 115 GB). Vaani's paper (arXiv 2603.28714) describes 80/10/10 within each district, with no speaker in two splits. M0 sampled from `ARTPARK-IISc/Vaani`, which has only `train`. M1 draws its frozen test sets from the official test split.
- **How Vaani was transcribed.** The vendors were Shaip, Megdap and Karya, using guidelines adapted from Bhashini, with transcribers from the same district. A language expert reviews each batch against a 10% issue threshold. No accuracy figure is published.

**Train against held-out clips.** Every M0 clip is also in `Vaani-transcription-part`, matched by audio file name with an identical transcript. 967 are in the official train split, 261 in validation and 283 in test (`data/manifests/m0_splits.jsonl`, from `data/splits.py`). That is close to the 64/18/18 of the whole Hindi split, not the paper's 80/10/10. A model trained on the train split should do better on those clips than on held-out ones (validation and test), and models that weren't should do the same on both. The same clips go to every model, so comparing the models' ratios cancels how hard the clips are. Intervals resample speakers.

| Model | WER, train | WER, held out | Train ÷ held out | Lenient, train ÷ held out |
|---|---|---|---|---|
| IndicConformer | 12.8% | 14.2% | 0.90 (0.81–1.01) | 0.89 (0.78–1.04) |
| Omnilingual 3B | 18.2% | 19.2% | 0.95 (0.85–1.05) | 0.95 (0.84–1.07) |
| Whisper-large-v3 | 25.6% | 26.4% | 0.97 (0.89–1.07) | 0.98 (0.89–1.09) |

- **IndicConformer gains a little more on train clips than the others, and the gap is too small to be sure of.** Against Whisper its ratio is 0.93 (0.86–1.01), or 0.91 (0.83–1.00) lenient. Against Omnilingual it's 0.95 (0.88–1.04). If the gap is real, it's worth about a point of WER on train clips.
- **IndicConformer's lead doesn't depend on it.** On held-out clips alone it scores 14.2%, against 19.2% for Omnilingual and 26.4% for Whisper.
- **This check is blind to training on the main Vaani release**, which holds train, validation and test clips alike. Vaani-Benchmark-V1.0 isn't: it was published in June 2026, after IndicConformer's release in March 2025, and isn't in either repository. Its Hindi set has 5,050 segments from 1,062 speakers in 102 districts (10.9 hours). Running all three models on it and comparing each model's benchmark WER with its M0 WER would settle contamination. Its three independent transcriptions per segment would also show how much transcribers disagree in each district, without a listener.

**Still open for M0:** the listening audit, which has 300 clips picked and its scoring ready (`audit/audit.py score`) but needs a Hindi-speaking listener. Also open: the three models on Vaani-Benchmark-V1.0's Hindi set, which needs a Kaggle run.
