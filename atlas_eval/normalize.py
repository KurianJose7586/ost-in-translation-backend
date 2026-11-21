"""Text normaliser applied to references and hypotheses before scoring.

Without it, spelling variants count as recognition errors. Rules, in order;
every rule has examples in tests/normalize_pairs.tsv:

1. Unicode NFC. Precomposed nukta letters (क़ U+0958 to य़ U+095F) are
   composition exclusions, so NFC decomposes them and both spellings meet.
2. Lowercase. Only Latin-script English changes.
3. Delete ZWJ, ZWNJ and the byte order mark, which change rendering, not
   words. A zero-width space marks a word break, so it becomes a space.
4. Delete the nukta (ज़ = ज), except in ड़ and ढ़, which are distinct letters
   and written consistently.
5. Chandrabindu becomes anusvara (माँ = मां).
6. A nasal consonant with virama before a consonant of its own class becomes
   anusvara (हिन्दी = हिंदी, सम्बन्ध = संबंध).
7. Devanagari digits become ASCII digits (२०२४ = 2024).
8. Apostrophes are deleted (don't = dont). Every other punctuation mark or
   symbol becomes a space (। ॥ , ? - " ₹).
9. Whitespace is collapsed.

Vowel signs and the virama are kept. Whisper's BasicTextNormalizer turns them
into spaces, which breaks Devanagari words apart; don't use it here.

Still open until Vaani's transcription conventions are read (KICKOFF A2):
numbers written as digits vs words, English written in Devanagari vs Latin,
and tags for noise and unclear speech.
"""

import re
import unicodedata

_NASALS = {"ङ": "कखगघ", "ञ": "चछजझ", "ण": "टठडढ", "न": "तथदध", "म": "पफबभ"}
_NASAL_RES = [re.compile(f"{n}्(?=[{c}])") for n, c in _NASALS.items()]
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalise(text: str) -> str:
    s = unicodedata.normalize("NFC", text).lower()
    s = re.sub("[‌‍﻿]", "", s).replace("​", " ")
    s = re.sub("(?<![डढ])़", "", s)
    s = s.replace("ँ", "ं")
    for r in _NASAL_RES:
        s = r.sub("ं", s)
    s = s.translate(_DIGITS)
    s = re.sub("['’]", "", s)
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return " ".join(s.split())
