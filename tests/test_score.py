from atlas_eval.score import align, build


def clip(i, district, speaker, text):
    return i, {"id": i, "text": text, "language": "Hindi", "district": district, "state": "S", "speaker": speaker}


def test_align():
    assert align("a b c d", "a x c") == [["C", "a", "a"], ["S", "b", "x"], ["C", "c", "c"], ["D", "d", None]]
    assert align("a b", "a b c") == [["C", "a", "a"], ["C", "b", "b"], ["I", None, "c"]]
    assert align("a b", "") == [["D", "a", None], ["D", "b", None]]


def test_build():
    clips = dict([clip("1", "A", "s1", "a b c d"), clip("2", "A", "s2", "e"), clip("3", "B", "s3", "a b")])
    hyps = {"m": {"1": "a x c", "2": "e", "3": "a b"}, "n": {"1": None, "2": None}}
    a, b = build(clips, hyps)["districts"]

    m = a["results"]["m"]
    assert m["wer"] == 2 / 5  # total errors over total words, not the mean of clip WERs (0.25)
    assert m["ci"][0] <= m["wer"] <= m["ci"][1]
    assert (m["clips"], m["speakers"]) == (2, 2)
    assert a["results"]["n"] == {"status": "unsupported"}
    assert "n" not in b["results"]  # not run on B at all

    # A (0.4, noisy) is pulled toward the language mean (0.2); B (0.0, no spread) isn't.
    assert 0.2 < m["wer_pooled"] < 0.4
    assert b["results"]["m"]["wer_pooled"] == 0.0
    assert [e["id"] for e in a["examples"]] == ["1", "2"]
