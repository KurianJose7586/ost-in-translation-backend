import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "data"))
from m1 import capped, parse, usable  # noqa: E402


def clip(speaker, district="Patna", start=0, seconds=5, text="एक घर है", session="0517330000"):
    name = (f"IISc_VaaniProject_M_Bihar_{district}_{speaker}_{session}_{district}-SPECIFIC_00012_"
            f"{start}_{start + seconds * 1000}.wav")
    return {"name": name, "state": "Bihar", "district": district, "transcript": text}


def test_parse():
    assert parse(clip("Chan39511")["name"]) == ("BIPA_Chan39511", "Patna-SPECIFIC_00012", 0, 5.0)
    assert parse("IISc_VaaniProject_S_Bihar_WestChamparan_127009_12363576_GENERIC_0869_5335_10486.wav") == (
        "BIWE_127009", "GENERIC_0869", 5335, 5.151)
    assert parse("IISc_VaaniProject_S_Bihar_Patna_NA_12363576_GENERIC_0869_5335_10486.wav") is None
    assert parse("not a vaani name.wav") is None


def test_usable():
    clips = [clip("a", start=i, session=str(i)) for i in range(10)]  # one speaker over ten sessions, capped at 4
    clips += [clip("b"), clip("b", start=1, seconds=40), clip("b", start=2, text="<noise></noise>")]  # 40 s and empty
    clips += [clip("c"), clip("c", district="Gaya")]  # different districts give different speakerIDs
    clips += [clip("d"), clip("d", district="Paschim")]  # one speakerID (BIPA_d) in two districts: dropped
    ok = usable(clips)
    assert {s: len(v) for s, v in ok["Bihar", "Patna"].items()} == {"BIPA_a": 10, "BIPA_b": 1, "BIPA_c": 1}
    assert list(ok["Bihar", "Gaya"]) == ["BIGA_c"]
    assert ok["Bihar", "Paschim"] == {}
    assert capped(ok["Bihar", "Patna"]) == 6
