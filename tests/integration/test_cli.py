"""End-to-end coverage for the PI-engine experiment command line."""

from __future__ import annotations

import os
import subprocess
import sys


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


def test_cli_reports_cutoff_safe_convergence_predictions() -> None:
    """Skipping registry, analyses, or evidence labels would make this fail."""
    result = _invoke("run", "linear_convergence", "--horizon", "3")

    assert result.returncode == 0, result.stderr
    assert "PI-ENGINE PREDICTION REPORT" in result.stdout
    assert "Prediction cutoff: 2026-08-08T12:00:00+00:00" in result.stdout
    assert "Report provenance:" in result.stdout
    assert "Model disagreement: retained separately" in result.stdout
    assert "synthetic-linear-affine-convergence@1.0.0: applicable" in result.stdout
    assert "Held-out scores: NOT REVEALED" in result.stdout
    assert "Calibration information:" in result.stdout


def test_cli_reveals_and_scores_convergence_and_negative_control() -> None:
    """Bypassing held-out scoring or control execution would make this fail."""
    convergence = _invoke("reveal", "linear_convergence", "--horizon", "3")
    control = _invoke("report", "irrelevant_proximity_control", "--horizon", "1")

    assert convergence.returncode == 0, convergence.stderr
    assert "Held-out scores: REVEALED" in convergence.stdout
    assert "mean_absolute_error=0.0" in convergence.stdout
    assert "Calibration information:" in convergence.stdout

    assert control.returncode == 0, control.stderr
    assert "Case: synthetic-irrelevant-proximity-control" in control.stdout
    assert "UNIDENTIFIABLE FROM AVAILABLE EVIDENCE" in control.stdout
    assert "Held-out scores: NOT REVEALED" in control.stdout
