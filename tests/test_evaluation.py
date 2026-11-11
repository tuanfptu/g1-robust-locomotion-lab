import json
from argparse import Namespace

import pytest

from g1lab.evaluation import evaluate
from g1lab.report import report


def test_short_protocol_exports_and_report(tmp_path):
    result = tmp_path / "eval"
    evaluate(
        Namespace(
            output=str(result), bundle=None, suite="flat", episodes=1, seconds=0.04, seed=10000, video=False
        )
    )
    summary = json.loads((result / "summary.json").read_text())
    assert len(summary) == 5
    assert summary["stand"]["episodes"] == 1
    report(Namespace(inputs=[str(result)], output=str(tmp_path / "report")))
    assert (tmp_path / "report" / "comparison.png").stat().st_size > 1000


def test_report_rejects_mismatched_protocols(tmp_path):
    for index, seed in enumerate((1, 2)):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "summary.json").write_text("{}")
        (folder / "protocol.json").write_text(
            json.dumps(dict(suite="flat", episodes=1, seconds=1, seed=seed))
        )
    with pytest.raises(ValueError, match="seed"):
        report(Namespace(inputs=[str(tmp_path / "0"), str(tmp_path / "1")], output=str(tmp_path / "report")))
