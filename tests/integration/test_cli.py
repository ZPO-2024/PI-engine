"""End-to-end coverage for the PI-engine experiment command line."""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path


def _invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_root = os.path.abspath("src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH")) if part
    )
    return subprocess.run(
        [sys.executable, "-m", "pi_engine.cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_cli_lists_synthetic_systems_and_negative_controls() -> None:
    """Omitting a catalog family would hide an available experiment case."""
    result = _invoke("list")

    assert result.returncode == 0, result.stderr
    assert "linear_convergence" in result.stdout
    assert "random_graph_control" in result.stdout
    assert "negative_control" in result.stdout


def test_cli_runs_then_reports_and_reveals_the_saved_convergence_artifact(
    tmp_path: Path,
) -> None:
    """Resimulating instead of consuming the saved artifact would fail here."""
    artifact = tmp_path / "linear-run.json"
    run = _invoke(
        "run", "linear_convergence", "--horizon", "3", "--artifact", str(artifact)
    )
    report = _invoke("report", "--artifact", str(artifact))
    reveal = _invoke("reveal", "--artifact", str(artifact))

    assert run.returncode == 0, run.stderr
    assert artifact.exists()
    assert "Run artifact saved:" in run.stdout
    assert report.returncode == 0, report.stderr
    assert "PI-ENGINE PREDICTION REPORT" in report.stdout
    assert "Prediction cutoff: 2026-08-08T12:00:00+00:00" in report.stdout
    assert "Report provenance:" in report.stdout
    assert "Model disagreement: retained separately" in report.stdout
    assert "synthetic-linear-affine-convergence@1.0.0: applicable" in report.stdout
    assert "Held-out scores: NOT REVEALED" in report.stdout
    assert reveal.returncode == 0, reveal.stderr
    assert "Held-out scores: REVEALED" in reveal.stdout
    assert "mean_absolute_error=0.0" in reveal.stdout
    assert "Calibration information:" in reveal.stdout


def test_cli_runs_reveals_and_scores_the_divergence_negative_control(
    tmp_path: Path,
) -> None:
    """Leaving the negative fixture unexecutable would make this fail."""
    artifact = tmp_path / "divergence-run.json"
    run = _invoke(
        "run", "deterministic_divergence", "--horizon", "3", "--artifact", str(artifact)
    )
    report = _invoke("report", "--artifact", str(artifact))
    reveal = _invoke("reveal", "--artifact", str(artifact))

    assert run.returncode == 0, run.stderr
    assert report.returncode == 0, report.stderr
    assert "Case: synthetic-deterministic-divergence" in report.stdout
    assert "Synthetic fixture regime: divergence" in report.stdout
    assert reveal.returncode == 0, reveal.stderr
    assert "Held-out scores: REVEALED" in reveal.stdout
    assert "mean_absolute_error=0.0" in reveal.stdout


def test_cli_rejects_a_tampered_run_artifact(tmp_path: Path) -> None:
    """Accepting changed artifact content would sever report provenance."""
    artifact = tmp_path / "tampered-run.json"
    run = _invoke(
        "run", "linear_convergence", "--horizon", "3", "--artifact", str(artifact)
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["payload"]["horizon"] = 1
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    report = _invoke("report", "--artifact", str(artifact))

    assert run.returncode == 0, run.stderr
    assert report.returncode != 0
    assert "run artifact integrity check failed" in report.stderr


def test_cli_retains_stochastic_member_residuals_after_reveal(tmp_path: Path) -> None:
    """Dropping ensemble members would hide residual evidence for raw samples."""
    artifact = tmp_path / "stochastic-run.json"
    run = _invoke(
        "run", "stochastic_branching", "--horizon", "4", "--artifact", str(artifact)
    )
    manifest = json.loads(artifact.read_text(encoding="utf-8"))
    ensemble = next(
        item["data"]
        for item in manifest["payload"]["artifacts"]
        if item["kind"] == "trajectory_ensemble"
    )
    reveal = _invoke("reveal", "--artifact", str(artifact))

    assert run.returncode == 0, run.stderr
    assert reveal.returncode == 0, reveal.stderr
    assert "Residual status:" in reveal.stdout
    for member in ensemble["trajectories"]:
        assert member["trajectory_id"] in reveal.stdout
