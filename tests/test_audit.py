import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "audit"))
from audit import disputed  # noqa: E402


def test_disputed():
    assert disputed("a b c", ["a x c"] * 3) == {1}  # all three wrote the same other word
    assert disputed("a b c", ["a c"] * 3) == {1}  # all three dropped it
    assert disputed("a b c", ["a x c", "a x c", "a y c"]) == set()  # they disagree
    assert disputed("a b c", ["a x c", "a x c", "a b c"]) == set()  # one heard the reference
