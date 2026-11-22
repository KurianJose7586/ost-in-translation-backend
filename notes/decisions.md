# Decisions

Newest first. Bigger planning decisions are in PLAN.md and FEASIBILITY.md, which live outside this repository.

| Date | Decision | Why |
|---|---|---|
| 23 Nov 2026 | *Open:* whether M1 test sets must be disjoint from Vaani's train split by image. Either drop test clips whose image is in train, or keep them and exclude the frozen test sets' images from Kaan's training data | The official split keeps speakers apart but not images: 80% of the Hindi draft's clips share an image with train. Dropping them leaves 67 Hindi districts with 100 clips, against 93 |
| 23 Nov 2026 | M1 test sets take at most 4 clips per speaker, not M0's 8 | The official test split holds every clip of a speaker, so 8 leaves a median of 27 speakers per Hindi district. 4 gives 42, for 93 qualifying districts instead of 99 |
| 23 Nov 2026 | A speaker is Vaani's speakerID, rebuilt from the audio file name: the state's and district's first two letters, then the name's speaker part (Patna: `BIPA_…`). The part after it is a recording session | The split repository has no speakerID column. The rule reproduces all 793 M0 speaker hashes; counting speaker and session together would give 102,825 Hindi test speakers instead of 11,481 |
| 23 Nov 2026 | An M1 clip id is speaker + image + start time in ms | The split repository has no utterance number |
| 22 Nov 2026 | Frozen test sets come from Vaani's official test split (`ARTPARK-IISc/Vaani-transcription-part`) | The plan asks for the official split. Clips with a known split also allow a train-versus-test contamination check |
| 22 Nov 2026 | Targets against a baseline are relative (EXPECTED_RESULTS.md) | The best open model is at 10–16% WER on M0. Point targets written for about 35% would ask for near-perfect output |
| 22 Nov 2026 | Digits and Latin-script English still count as errors | Under 3% of Whisper's M0 errors and none of the other models'. A fix means maintaining a lexicon |
| 22 Nov 2026 | The audit corrects each district's full WER by the gap it measures, rather than rescoring the 30 audited clips | The gap varies far less than a 30-clip WER. In Dhar, IndicConformer's WER on 15 random clips was 2.7%, against 5.3% on the 116 they were drawn from |
| 22 Nov 2026 | Listening audit: per district, 15 clips where all three models agree against the reference plus 15 random others, each tagged with its stratum's size | Suspects find reference errors fast; the random half and the stratum sizes let the error rate be weighted back to the whole district |
| 21 Nov 2026 | Report lenient WER next to WER | About a third of a strong model's M0 errors were spelling-only (है/हैं, बहोत/बहुत). Any claim about Kaan must hold on both, because a model trained on Vaani learns Vaani's spellings |
| 21 Nov 2026 | Runner scripts use Python 3.12, PyAV below 19, torchaudio 2.8 and pyarrow below 26 for Omnilingual, and their own CUDA 12 cuBLAS and cuDNN for Whisper | Each fixed a failure on Kaggle's Python 3.13 image |
| 21 Nov 2026 | Every step of a run resumes where it stopped | Kaggle sessions die; a rerun redoes only missing clips |
| 21 Nov 2026 | Speaker IDs in published manifests are hashed | Some Vaani speaker IDs embed fragments of names |
| 21 Nov 2026 | A clip id is speaker + image + utterance number | Vaani's utterance numbers restart for each image |
| 21 Nov 2026 | Clips with speaker ID `NA` never enter a test set | Speaker-disjoint splits and speaker bootstrap need real IDs |
| 21 Nov 2026 | Test slice: at most 8 clips per speaker, under 40 s, sampled from random row groups | Many speakers per district, and Omnilingual's 40-second limit |
| 21 Nov 2026 | Read Vaani by file footer and row group, never whole files | Vaani Hindi alone is 5,620 files, and the Hub allows 5,000 requests per five minutes |
| 21 Nov 2026 | M0 language is Hindi, in ten districts across five dialect regions | Hindi is the only Hindi-belt language with ten or more transcribed districts (113). Dialect labels are thin |
| 21 Nov 2026 | Districts created since 2011 are drawn inside their 2011 parent polygon | The open boundary map is Census 2011 |
| 21 Nov 2026 | Licences: Vaani and IndicVoices CC BY 4.0, FLEURS CC BY 4.0, DataMeet district boundaries CC BY 2.5 India, IndicConformer MIT | Checked on each dataset or model page |
