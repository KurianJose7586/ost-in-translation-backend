"""Let a runner pick up where an interrupted run stopped.

    python runners/resume.py CLIPS OUT_DIR [--no-audio] > todo.jsonl

As a script, it prints the clips that still need audio: some model supports the
clip's language and its transcripts in OUT_DIR (whisper.jsonl, omnilingual.jsonl,
indicconformer.jsonl) don't have the clip yet. A chained Kaggle run fetches only those.
With --no-audio it prints the clips that still need a line only from models that don't
support them (hyp null), which the runners write without opening the audio.
"""

import json
import os
import sys


def finished(path):
    """Drop a half-written last line from OUT and return the ids it already holds."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.endswith("\n")]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return {json.loads(line)["id"] for line in lines}


def failed(e):
    """Transcript fields for a clip the model couldn't read.

    One unreadable file would otherwise stop the runner, and every resumed run after it,
    at the same clip. The clip is marked done with its error, and atlas-eval leaves it out
    for every model. A transient error (out of GPU memory) is kept too: delete those lines
    to retry them.
    """
    return {"hyp": "", "error": f"{type(e).__name__}: {e}"[:300]}


if __name__ == "__main__":
    from langs import INDICCONFORMER, OMNILINGUAL, WHISPER

    clips, out_dir = sys.argv[1:3]
    maps = {"whisper": WHISPER, "omnilingual": OMNILINGUAL, "indicconformer": INDICCONFORMER}
    done = {m: finished(os.path.join(out_dir, f"{m}.jsonl")) for m in maps}
    with open(clips, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lacking = [m for m in maps if r["id"] not in done[m]]
            audio = any(r["language"] in maps[m] for m in lacking)
            if lacking and audio != ("--no-audio" in sys.argv):
                sys.stdout.write(line)
