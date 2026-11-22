"""Which language each baseline is told, keyed by Vaani's language names.

Varieties a model doesn't know are forced to the closest language it does,
usually Hindi, because that is what a real user would pick. A language missing
from a map means the model doesn't support it: its runner writes hyp=null and
the atlas shows "unsupported" instead of a score. A model counts as supporting a
variety only if it writes the script Vaani transcribes it in: Vaani writes Chakma,
Garo, Mizo, Nagamese and Wancho in Latin script, and Tulu in Kannada script.
Publish this on the methodology page.

Covers every language in the M1 test sets (data/manifests/m1_*.jsonl).
"""

HINDI_BELT = [
    "Hindi", "Bhojpuri", "Maithili", "Magahi", "Awadhi", "Bundeli", "Bagheli", "Chhattisgarhi",
    "Marwari", "Haryanvi", "Angika", "Bajjika", "Khortha", "Khariboli", "Malvi", "Nimadi", "Mewari",
    "Mewati", "Jaipuri", "Shekhawati", "Harauti", "Rajasthani", "Garhwali", "Kumaoni", "Surgujia",
    "Sadri", "Wagdi", "Halbi",
]

WHISPER = {
    **dict.fromkeys(HINDI_BELT, "hi"),
    "Urdu": "ur", "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Tamil": "ta",
    "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Assamese": "as", "Nepali": "ne",
    "Sindhi": "sd", "English": "en",
    "Tulu": "kn", "Konkani": "mr",  # no Tulu or Konkani; the closest language in the same script
}

# Codes checked against omnilingual_asr.models.wav2vec2_llama.lang_ids.supported_langs.
OMNILINGUAL = {
    **dict.fromkeys(HINDI_BELT, "hin_Deva"),
    "Bhojpuri": "bho_Deva", "Maithili": "mai_Deva", "Magahi": "mag_Deva", "Awadhi": "awa_Deva",
    "Bundeli": "bns_Deva", "Bagheli": "bfy_Deva", "Chhattisgarhi": "hne_Deva", "Marwari": "rwr_Deva",
    "Haryanvi": "bgc_Deva", "Angika": "anp_Deva", "Garhwali": "gbm_Deva", "Malvi": "mup_Deva",
    "Wagdi": "wbr_Deva", "Sadri": "sck_Deva",
    "Urdu": "urd_Arab", "Bengali": "ben_Beng", "Marathi": "mar_Deva", "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru", "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym", "Assamese": "asm_Beng", "Nepali": "nep_Deva", "Odia": "ory_Orya",
    "English": "eng_Latn",
    "Halbi": "hlb_Deva", "Konkani": "gom_Deva", "Mizo": "lus_Latn", "Nagamese": "nag_Latn",
    "Tulu": "kan_Knda",  # its Tulu is tcy_Mlym, in Malayalam script
    # Garo is grt_Beng only (Bengali script); no Chakma or Wancho.
}

INDICCONFORMER = {
    **dict.fromkeys(HINDI_BELT, "hi"),
    "Maithili": "mai",
    "Urdu": "ur", "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Tamil": "ta",
    "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Assamese": "as", "Nepali": "ne",
    "Sindhi": "sd", "Odia": "or", "Konkani": "kok", "Kashmiri": "ks", "Dogri": "doi",
    "Manipuri": "mni", "Santali": "sat",
    "Tulu": "kn",
}
