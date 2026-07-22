"""End-to-end --dry-run: mock backends + bundled fixtures, generate->judge->analyze."""
import argparse
from pathlib import Path

from src.pipeline import run_dry_run


def _dry_run_args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(tmp_path), run_id="dry-run",
        judge_mode=None, decomposer_spec=None,
    )


def test_dry_run_produces_all_outputs(tmp_path):
    run_dry_run(_dry_run_args(tmp_path))
    run_dir = Path(tmp_path) / "runs" / "dry-run"
    for name in ("config.json", "generations.jsonl", "decompositions.jsonl",
                 "verifications.jsonl", "analysis.md", "curves.png",
                 "raw_summary.csv"):
        assert (run_dir / name).exists(), f"missing {name}"
    assert (run_dir / "curves.png").stat().st_size > 1000
    md = (run_dir / "analysis.md").read_text()
    assert "Effective Faithful Context" in md
    assert "Cohen's kappa" in md  # two judges in the dry run


def test_dry_run_resume_is_free(tmp_path):
    run_dry_run(_dry_run_args(tmp_path))
    run_dir = Path(tmp_path) / "runs" / "dry-run"
    gens = (run_dir / "generations.jsonl").read_text()
    verifs = (run_dir / "verifications.jsonl").read_text()

    run_dry_run(_dry_run_args(tmp_path))  # second pass over the same run id
    assert (run_dir / "generations.jsonl").read_text() == gens  # no new records
    assert (run_dir / "verifications.jsonl").read_text() == verifs
