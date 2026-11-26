# Results

Newest first. Every number here can be rebuilt with `atlas-eval` from the raw transcripts. Those live outside git, in `runs/`, until a round is frozen.

## Vaani-Benchmark-V1.0, 24 Nov 2026: the held-out contamination check (Hindi)

**Setup.** The Hindi set of `ARTPARK-IISc/Vaani-Benchmark-V1.0`: 5,050 segments from 1,062 speakers in 102 districts, 10.9 hours, none 40 s or longer. Each segment has three independent transcriptions. The set was published after IndicConformer and is in neither Vaani repository. The three M0 models ran with M0's settings on Kaggle (`runners/kaggle_bench.sh`), and `data/benchmark.py score` scored them in two ways:

- **The benchmark's own WER** is approach 3 of Vaani's evaluation toolkit. A word counts as an error only if no transcription has it right, and a deletion only if all three transcriptions delete it.
- **Our WER** is against transcription 1 with `normalise()`, as on M0.

Intervals are 95%, from 1,000 speaker resamples. Omnilingual finished 3,264 segments before the week's Kaggle GPU hours ran out. Those are the first in file order, not a random sample, so its numbers cover those segments only. Its last 1,786 segments are still to run.

All 5,050 segments:

| Model | Benchmark's own WER | Our WER | Lenient |
|---|---|---|---|
| IndicConformer | 13.2% | 16.2% (15.6–17.0) | 12.0% |
| Whisper-large-v3 | 26.0% | 29.6% (28.5–30.7) | 26.0% |
| Transcription 2 | | 8.7% (8.2–9.4) | 7.0% |
| Transcription 3 | | 11.1% (10.5–11.7) | 8.7% |

The 3,264 segments Omnilingual finished (694 speakers):

| Model | Benchmark's own WER | Our WER | Lenient |
|---|---|---|---|
| IndicConformer | 13.8% | 17.2% (16.4–18.0) | 12.8% |
| Omnilingual 3B | 17.8% | 21.3% (20.4–22.3) | 17.5% |
| Whisper-large-v3 | 27.2% | 31.3% (29.9–32.6) | 27.6% |
| Transcription 2 | | 10.1% (9.4–10.8) | 7.9% |
| Transcription 3 | | 11.8% (11.1–12.5) | 9.3% |

**The contamination check.** A model trained on Vaani's main release has heard M0's speakers but not the benchmark's, so it should lose more than the others going from M0 to the benchmark. The benchmark is harder for every model, so compare each model's ratio of benchmark WER to M0 WER (our WER both times), and then the ratios against each other. That cancels what the two sets do to every model alike, though not what they do to one model alone.

| Model | Segments | M0 WER | Benchmark WER | Benchmark ÷ M0 | Lenient, benchmark ÷ M0 |
|---|---|---|---|---|---|
| IndicConformer | 5,050 | 13.3% | 16.2% | 1.22 (1.15–1.30) | 1.28 (1.18–1.40) |
| Whisper-large-v3 | 5,050 | 25.9% | 29.6% | 1.14 (1.08–1.21) | 1.18 (1.11–1.26) |
| IndicConformer | 3,264 | 13.3% | 17.2% | 1.29 (1.21–1.39) | 1.37 (1.25–1.50) |
| Omnilingual 3B | 3,264 | 18.6% | 21.3% | 1.15 (1.07–1.23) | 1.19 (1.10–1.29) |
| Whisper-large-v3 | 3,264 | 25.9% | 31.3% | 1.21 (1.14–1.28) | 1.25 (1.17–1.35) |

**Findings:**

- **The pipeline reproduces the published number.** IndicConformer scores 13.2% on the benchmark's own measure, exactly its published Hindi WER. So the fetch, the runner and the scoring match the benchmark's.
- **IndicConformer loses more than the others on held-out speakers. The gap is small, and it is real.**
  - Its ratio against Whisper's is 1.07 (1.02–1.12), or 1.08 (1.03–1.15) lenient. Against Omnilingual's, on the segments both have, it is 1.13 (1.08–1.18).
  - That fits some Vaani in IndicConformer's training data. It agrees with the train against held-out check on M0, where IndicConformer's ratio against Whisper's was 0.93 (0.86–1.01).
  - It is worth about a point of WER. Had IndicConformer lost what Whisper loses, it would score about 15.2% here, not 16.2%, so M0 flatters it by roughly one point.
  - Region explains a little of it. M0's districts are all in Hindi-belt states, and about half the benchmark's aren't. On the benchmark's Hindi-belt states alone (2,589 segments), the ratio against Whisper's is 1.06 (1.01–1.11), or 1.06 (1.00–1.13) lenient. Against Omnilingual's, on the shared segments there, it is 1.10 (1.05–1.15). The gap shrinks but stays.
  - The two sets also differ in how the speech was segmented and in which batches transcribed it, which a ratio of ratios can't rule out. Only AI4Bharat can say what IndicConformer was trained on, so ask them (vaani_email_draft.md).
- **IndicConformer's lead survives.** It scores 16.2% against Whisper's 29.6% on the whole set, and 17.2% against Omnilingual's 21.3% on the shared segments.
- **Omnilingual shows no sign of it.** It loses slightly less than Whisper: 0.95 (0.91–0.99), or 0.95 (0.90–1.00) lenient. Those are the first 3,264 segments only, which are harder than the rest: Whisper scores 31.3% on them and 29.6% on all 5,050.
- **Human transcribers disagree by 9–11%.**
  - Transcriptions 2 and 3 score 8.7% and 11.1% against transcription 1, or 7.0% and 8.7% lenient.
  - No model can meaningfully score below that against a single transcription. IndicConformer's 16.2% is 5.1 to 7.5 points above it.
- **Districts where transcribers disagree are only a little harder for the models.** (Corrected 26 Nov; the first version of this section said 0.42.)
  - Disagreement is measured as transcription 2 scored against transcription 3. Across the 102 districts, its rank correlation with WER is 0.23 for IndicConformer and 0.24 for Whisper. On the shared segments it is 0.26, 0.38 for Omnilingual and 0.29 for Whisper.
  - Scoring transcriptions 2 and 3 against transcription 1 instead gives 0.42 and 0.40. That is inflated: the models' WER is scored against transcription 1 too, so a district whose transcription 1 is poor raises both. About half of the apparent link is errors in transcription 1, not speech that people find ambiguous.
  - `data/benchmark.py score` also prints each district's three-reference WER, which a poor transcription 1 can't raise.
  - Among districts with at least 50 segments, IndicConformer does worst in Sarguja, Chhattisgarh, at 22.8%, then Bihar's Jamui, Gopalganj and Bhagalpur, at about 17%. It does best in Madhya Pradesh: Dhar 5.6%, Katni 7.0% and Bhopal 7.9%.
- **District rankings are only moderately reliable at the benchmark's sample sizes.**
  - Split-half reliability is the rank correlation between the district WERs from two random halves of each district's speakers, averaged over 200 splits and corrected to full size (Spearman–Brown). It is 0.49 for IndicConformer and 0.45 for Whisper across the 102 districts; 38 of them have fewer than 30 segments.
  - On M0's ten districts, with about 150 clips each, it is 0.67 for IndicConformer, 0.82 for Omnilingual and 0.82 for Whisper (`atlas-eval` prints it for every round).
  - Four M0 districts are also in the benchmark, with other speakers and transcribers (`benchmark.py score --retest runs/m0/atlas.json`):

    | District | Benchmark segments | IndicConformer, M0 | IndicConformer, benchmark | Whisper, M0 | Whisper, benchmark |
    |---|---|---|---|---|---|
    | Kabirdham | 26 | 16.1% | 22.6% | 31.9% | 51.7% |
    | Rohtak | 110 | 16.1% | 9.8% | 29.7% | 17.2% |
    | Katni | 105 | 11.5% | 7.0% | 20.1% | 16.8% |
    | Dhar | 105 | 12.9% | 5.6% | 22.6% | 14.3% |

  - Kabirdham stays the hardest, on only 26 segments. Rohtak, second-hardest on M0 and the district behind the lenient gate for all three models, is one of the easier districts on the benchmark. Four districts can't settle whether district ranks replicate. M1's Hindi districts against the benchmark's will (`--retest` on M1's round), and the M0 gate stays provisional until the listening audit.
- **IndicConformer returned nothing for 12 segments.** All 12 are noisy: every reference opens with a noise tag (`<noise>`, `<static noise>`, `<bird sound>`), and on the same segments Whisper invents text, including its familiar "सब्सक्राइब करें". They count as all deletions, which is fair: the speech is there.
- **Kaggle speeds, for planning M1:**
  - Whisper took 6.2 hours for all 5,050 segments on a T4.
  - Omnilingual took 7–12 s a segment on the other T4, probably because its default bfloat16 isn't native on a T4. At that pace, M1's 28,370 clips would take it 55 to 95 hours.
  - IndicConformer took 2.4 hours on the CPU alone, which uses no GPU hours.

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
- **The official split keeps speakers apart, not images.** In Hindi, every generic test image and 73% of district-specific ones also appear in train. 80% of the Hindi draft's clips describe an image that train clips also describe. In the other large languages, 95–100% of generic-image clips and 20–79% of specific-image clips share their image with train. Dropping those clips would leave 67 Hindi districts with 100 clips, against 93.
- **Holding the draft's images out of training would cost 44.5% of Hindi's training clips** (`m1.py images Hindi`, 26 Nov). Of 424,964 train clips, 178,664 (42.0%) describe a generic image the Hindi draft uses and 10,625 (2.5%) a district-specific one. So the test sets stay as drafted. Kaan leaves out only the 2.5%, and its results are reported on generic-image and specific-image clips separately (decisions.md, 26 Nov).
- **Every qualifying district is on the map.** `data/vaani_districts.tsv` gained 16 districts that only the split repository has, mostly in Kerala, Tamil Nadu and Telangana.
- **40 M0 speakers are in the Hindi draft** (corrected 26 Nov from 22). M0 drew from all three splits, so that is expected; M0 isn't training data.
- **No speaker ID collides.** Rebuilding speaker IDs from the first two letters of state and district could give two people in one state the same ID, and `usable()` would drop both. Across all 64 test languages, none of the 23,191 speaker IDs comes from two different speakers.
- **Every clip now says whether its image is also in train** (`image_in_train`, 26 Nov): 21,866 of the 28,370 clips. The drafts are otherwise unchanged; the Hindi draft's SHA-256 is now `fa2e2bf0cd3e1efc0be9adab3cdd65d2e548fea2258bfa18605fe35d4b635b55`.

## M0, 22 Nov 2026: three baselines

**Setup.** Ten Hindi districts (`data/m0_districts.tsv`), 1,511 clips from 793 speakers (`data/manifests/m0.jsonl`), run on a Kaggle T4. Models: Whisper-large-v3 (faster-whisper), Omnilingual ASR `omniASR_LLM_3B_v2` with a Hindi hint, and IndicConformer 600M. WER uses `normalise()`, and lenient WER uses `lenient()`. Intervals are 95%, from 1,000 speaker resamples. Rows are ordered by the best model's WER, worst first. The raw transcripts are in `results/m0/`. On 26 Nov each district got its own bootstrap seed, so adding districts never moves another's interval; that moved these intervals by at most 0.5 points, and the gate below still holds.

**M0 can never evaluate Kaan.** 967 of its clips are in Vaani's official train split (below), which Kaan trains on. Compare Kaan with the baselines on M1 only.

| District | Whisper-large-v3 | Omnilingual 3B | IndicConformer |
|---|---|---|---|
| Kabirdham | 31.9% (29.5–34.2) | 23.4% (21.5–25.2) | 16.1% (14.7–17.9) |
| Rohtak | 29.7% (25.9–33.5) | 21.6% (18.1–25.2) | 16.1% (13.4–18.7) |
| West Champaran | 28.2% (23.7–33.4) | 20.5% (17.4–23.8) | 14.5% (12.2–17.1) |
| Dhar | 22.6% (18.9–26.2) | 15.0% (11.6–18.4) | 12.9% (10.4–15.4) |
| Patna | 23.7% (20.7–27.0) | 19.4% (15.7–23.3) | 12.6% (10.5–15.2) |
| Lucknow | 24.0% (21.5–26.3) | 16.0% (13.7–18.4) | 12.3% (10.3–14.4) |
| Garhwa | 22.2% (20.2–24.1) | 15.6% (13.5–17.7) | 11.8% (10.3–13.4) |
| Bikaner | 26.1% (23.8–28.5) | 18.4% (16.2–20.5) | 11.5% (10.1–13.3) |
| Katni | 20.1% (15.7–24.0) | 14.0% (10.7–17.3) | 11.5% (8.7–14.1) |
| Lalitpur | 21.7% (18.9–24.6) | 14.8% (12.4–17.3) | 10.2% (8.4–12.2) |
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
  - On Vaani-Benchmark-V1.0, IndicConformer's published Hindi WER is 13.2, against 13.3% here. The two can't be compared directly: the benchmark scores against the closest of three references, which lowers WER, and its clips and normalisation differ. (Running it ourselves on 24 Nov gave 13.2% on the benchmark's measure and 16.2% on ours: see the Vaani-Benchmark section.)
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
- **This check is blind to training on the main Vaani release**, which holds train, validation and test clips alike. Vaani-Benchmark-V1.0 isn't: it was published in June 2026, after IndicConformer's release in March 2025, and isn't in either repository. Its Hindi set has 5,050 segments from 1,062 speakers in 102 districts (10.9 hours). Running all three models on it and comparing each model's benchmark WER with its M0 WER would settle contamination. Its three independent transcriptions per segment would also show how much transcribers disagree in each district, without a listener. (Done on 24 Nov: see the Vaani-Benchmark section.)

**Still open for M0:** the listening audit, which has 300 clips picked and its scoring ready (`audit/audit.py score`) but needs a Hindi-speaking listener. Until then, the benchmark's three transcriptions are interim, listener-free evidence for Rohtak, Dhar and Katni (see the Vaani-Benchmark section). The three models on Vaani-Benchmark-V1.0's Hindi set are done, except Omnilingual's last 1,786 segments.
