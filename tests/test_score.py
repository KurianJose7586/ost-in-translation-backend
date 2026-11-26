import json

import numpy as np
import pytest

from atlas_eval.score import align, build, main, reliability


def clip(i, district, speaker, text, language="Hindi"):
    return i, {"id": i, "text": text, "language": language, "district": district, "state": "S", "speaker": speaker}


def test_align():
    assert align("a b c d", "a x c") == [["C", "a", "a"], ["S", "b", "x"], ["C", "c", "c"], ["D", "d", None]]
    assert align("a b", "a b c") == [["C", "a", "a"], ["C", "b", "b"], ["I", None, "c"]]
    assert align("a b", "") == [["D", "a", None], ["D", "b", None]]


def test_build():
    clips = dict([clip("1", "A", "s1", "a b c d"), clip("2", "A", "s2", "e"), clip("3", "B", "s3", "a b")])
    hyps = {"m": {"1": "a x c", "2": "e", "3": "a b"}, "n": {"1": None, "2": None}}
    out = build(clips, hyps)
    a, b = out["districts"]
    assert out["version"] == 2 and out["languages"] == ["Hindi"]

    m = a["results"]["Hindi"]["m"]
    assert m["wer"] == 2 / 5  # total errors over total words, not the mean of clip WERs (0.25)
    assert m["ci"][0] <= m["wer"] <= m["ci"][1]
    assert (m["clips"], m["speakers"]) == (2, 2)
    assert a["results"]["Hindi"]["n"] == {"status": "unsupported"}
    assert "n" not in b["results"]["Hindi"]  # not run on B at all

    # A (0.4, noisy) is pulled toward the language mean (0.2); B (0.0, no spread) isn't.
    assert 0.2 < m["wer_pooled"] < 0.4
    assert b["results"]["Hindi"]["m"]["wer_pooled"] == 0.0
    assert [e["id"] for e in a["examples"]["Hindi"]] == ["1", "2"]


def test_two_languages_in_one_district():
    # Pune-like: each language is scored on its own clips, and a model that supports one of
    # them is unsupported in the other instead of being scored on half the district.
    clips = dict([clip("1", "P", "s1", "a b", "Hindi"), clip("2", "P", "s2", "c d", "Marathi"), clip("3", "Q", "s3", "e", "Marathi")])
    hyps = {"m": {"1": "a b", "2": "x y", "3": "e"}, "w": {"1": "a x", "2": None, "3": None}}
    out = build(clips, hyps)
    p = out["districts"][0]
    assert p["languages"] == {"Hindi": 0.5, "Marathi": 0.5}
    assert p["results"]["Hindi"]["m"]["wer"] == 0.0 and p["results"]["Marathi"]["m"]["wer"] == 1.0
    assert p["results"]["Hindi"]["w"]["wer"] == 0.5 and p["results"]["Marathi"]["w"] == {"status": "unsupported"}
    assert out["languages"] == ["Marathi", "Hindi"]  # most clips first


def test_lenient_wer():
    # "हैं" heard as "है" is a strict error but a lenient match; a wrong word is an error in both.
    clips = dict([clip("1", "A", "s1", "ये घर हैं"), clip("2", "A", "s2", "ये घास है")])
    m = build(clips, {"m": {"1": "ये घर है", "2": "ये घर है"}})["districts"][0]["results"]["Hindi"]["m"]
    assert m["wer"] == 2 / 6 and m["wer_lenient"] == 1 / 6
    assert m["ci_lenient"][0] <= m["wer_lenient"] <= m["ci_lenient"][1]


def test_intervals_dont_move_when_a_district_is_added():
    base = dict(clip(str(i), "A", f"s{i}", "a b c") for i in range(6))
    hyps = {"m": {str(i): "a b" if i % 2 else "a b c" for i in range(6)}}
    before = build(base, hyps)["districts"][0]["results"]["Hindi"]["m"]["ci"]
    grown = base | dict([clip("9", "B", "s9", "x")])
    after = build(grown, {"m": hyps["m"] | {"9": "x"}})["districts"][0]["results"]["Hindi"]["m"]["ci"]
    assert before == after


def test_reliability():
    rng = np.random.default_rng(0)
    # Five districts at clearly different WERs, four speakers each: halves agree perfectly.
    ds = [np.array([[k, 10]] * 4, dtype=float) for k in range(5)]
    assert reliability(ds, rng) == pytest.approx(1.0)
    assert reliability(ds[:4], rng) is None  # too few districts


def test_main_refuses_partial_and_skips_failed_clips(tmp_path, capsys):
    rows = [{"id": i, "text": "a b", "language": "Hindi", "district": "A", "state": "S", "speaker": f"s{i}"} for i in "123"]
    (tmp_path / "m.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    (tmp_path / "x.jsonl").write_text("\n".join(json.dumps({"id": i, "model": "x", "hyp": "a b"}) for i in "12"), encoding="utf-8")
    args = [str(tmp_path / "m.jsonl"), str(tmp_path / "x.jsonl"), "-o", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit):
        main(args)
    main(args + ["--allow-partial"])
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["partial"] is True

    lines = [{"id": "1", "model": "x", "hyp": "a b"}, {"id": "2", "model": "x", "hyp": "", "error": "unreadable"}, {"id": "3", "model": "x", "hyp": "a"}]
    (tmp_path / "x.jsonl").write_text("\n".join(map(json.dumps, lines)), encoding="utf-8")
    main(args)
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    r = out["districts"][0]["results"]["Hindi"]["x"]
    assert r["clips"] == 2 and r["wer"] == 0.25 and "partial" not in out  # clip 2 left out, not scored as all deleted
    assert "1 clips a runner couldn't read" in capsys.readouterr().err
