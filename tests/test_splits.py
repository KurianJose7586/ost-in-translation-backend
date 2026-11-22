import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "data"))
from splits import speaker  # noqa: E402


def test_speaker_leaves_out_the_session():
    one = "IISc_VaaniProject_M_Bihar_Patna_{}_Patna-SPECIFIC_00012_1200_5400.wav"
    assert speaker(one.format("ab12cd_s1")) == speaker(one.format("ab12cd_s2")) == ("Bihar", "Patna", "ab12cd")
    assert speaker("IISc_VaaniProject_S_Bihar_Patna_ab12cd_s1_GENERIC_0042_0_900.wav")[2] == "ab12cd"
