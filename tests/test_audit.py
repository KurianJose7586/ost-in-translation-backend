import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "audit"))
from audit import disputed, estimate  # noqa: E402


def test_disputed():
    assert disputed("a b c", ["a x c"] * 3) == {1}  # all three wrote the same other word
    assert disputed("a b c", ["a c"] * 3) == {1}  # all three dropped it
    assert disputed("a b c", ["a x c", "a x c", "a y c"]) == set()  # they disagree
    assert disputed("a b c", ["a x c", "a x c", "a b c"]) == set()  # one heard the reference


def test_estimate():
    # 2 of 10 small-stratum clips and 3 of 90 large-stratum clips: each stands for 5 and 30 clips
    strata = [(10, np.array([[1.0], [3.0]])), (90, np.array([[0.0], [0.0], [3.0]]))]
    point, boot = estimate(strata, np.random.default_rng(0))
    assert point.tolist() == [5 * 4 + 30 * 3]
    assert boot.shape == (1000, 1) and boot.min() >= 0 and boot.max() <= 5 * 6 + 30 * 9
    same = [(10, np.array([[2.0], [2.0]]))]
    assert (estimate(same, np.random.default_rng(0))[1] == 20).all()  # nothing to redraw
