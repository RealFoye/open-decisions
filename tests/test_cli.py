import json

from opendecisions.calibration import Calibrator
from opendecisions.cli import main


def test_calibrate_command(tmp_path, capsys):
    rows = []
    for i in range(12):
        text = f"Ticket {i}: the parcel was {'late' if i % 2 else 'fine'}."
        rows.append({"input": text, "question": {"type": "noul", "instructions": "The parcel was late."},
                     "label": bool(i % 2)})
        rows.append({"input": text, "question": {"type": "choice", "instructions": "Mood?",
                                                 "options": {"angry": "", "calm": "", "happy": ""}},
                     "label": "angry" if i % 2 else "calm"})
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "cal.json"
    main(["calibrate", "--model", "tiny-random", "--data", str(data), "-o", str(out)])
    cal = Calibrator.load(out)
    assert "choice" in cal.temperature and "noul" in cal.temperature
    printed = capsys.readouterr().out
    assert "held-out after" in printed and "ece=" in printed
