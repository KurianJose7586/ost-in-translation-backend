import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "data"))
from benchmark import approach3_wer, clean_text  # noqa: E402


def test_clean_text():
    assert clean_text("टाइम {time}, <noise>हाँ</noise> [breathing] है।") == "टाइम हाँ है"
    assert clean_text("") == clean_text(None) == "NA"  # an empty prediction costs one error


def test_approach3_wer():
    assert approach3_wer(["a b c"], ["a x c"], ["a b c d"], ["a x c"]) == 0  # one transcriber heard x
    assert approach3_wer(["a b c"], ["a b c"], ["a b c"], ["a y c"]) == 1 / 3  # all three heard b
    assert approach3_wer(["a b"], ["a b"], ["a"], ["a"]) == 0  # deleted only if all three have the word
    assert approach3_wer(["a b"], ["a b"], ["a b"], ["a"]) == 1 / 2
