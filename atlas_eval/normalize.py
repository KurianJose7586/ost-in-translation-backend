"""Text normaliser applied to references and hypotheses before scoring.

Without it, spelling variants count as recognition errors. Rules, in order;
every rule has examples in tests/normalize_pairs.tsv:

0. Vaani's annotations go. Braces hold a gloss: the Latin spelling of an
   English word (फोटो {photo}) or the standard Hindi for a dialect word
   (इ {इस}); the spoken form before the braces is what gets scored. Square
   brackets hold non-speech events ([breathing]), and angle brackets hold
   noise and pause tags (<noise> ... </noise>, <pause>); the words inside stay.
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
8. Apostrophes and underscores are deleted (don't = dont; Vaani writes LED
   as एल_इ_डी, which becomes एलइडी). Every other punctuation mark or symbol
   becomes a space (। ॥ , ? - " ₹, and Vaani's -- for cut-off speech).
9. Whitespace is collapsed.

Vowel signs and the virama are kept. Whisper's BasicTextNormalizer turns them
into spaces, which breaks Devanagari words apart; don't use it here.

`lenient()` is the second metric. It applies the rules above, then also
ignores the nasal mark (है = हैं, में = मे) and maps a short list of variants
of the same word (बहोत = बहुत, the homophones कि = की). About a third of a
strong model's M0 errors were these. Reporting both shows whether a district's
score comes from hearing or from spelling conventions, and stops a model
trained on Vaani from winning just by learning Vaani's spellings. Add a pair
to the list only when both sides are the same word.

Still open: a model that writes an English word in Latin script ("photo")
scores an error against Vaani's Devanagari (फोटो). Vaani's braces give a
Latin-to-Devanagari lexicon that could fix this; decide after reading M0 errors.
"""

import re
import unicodedata

_NASALS = {"ङ": "कखगघ", "ञ": "चछजझ", "ण": "टठडढ", "न": "तथदध", "म": "पफबभ"}
_NASAL_RES = [re.compile(f"{n}्(?=[{c}])") for n, c in _NASALS.items()]
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Same word, different spelling: spoken forms Vaani writes down, homophones, a typo, an English loanword.
_SAME_WORD = {"बहोत": "बहुत", "बोहोत": "बहुत", "कि": "की", "हे": "है", "यह": "ये", "पे": "पर", "वाइट": "व्हाइट"}


def normalise(text: str) -> str:
    s = re.sub(r"\{[^}]*\}|\[[^\]]*\]|<[^>]*>", " ", text)
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub("[‌‍﻿]", "", s).replace("​", " ")
    s = re.sub("(?<![डढ])़", "", s)
    s = s.replace("ँ", "ं")
    for r in _NASAL_RES:
        s = r.sub("ं", s)
    s = s.translate(_DIGITS)
    s = re.sub("['’_]", "", s)
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return " ".join(s.split())


def lenient(text: str) -> str:
    """normalise(), then ignore the nasal mark and map variants of the same word."""
    return " ".join(_SAME_WORD.get(w, w) for w in normalise(text).replace("ं", "").split())
