"""Runner for the A4 negative-control analysis.

Solanezumab is a soluble-amyloid antibody that does not cause ARIA in the
way that anti-plaque mAbs (lecanemab, aducanumab, donanemab) do.  Running
the standard DR estimator on A4 solanezumab-vs-placebo therefore serves as
a *negative control exposure* — the estimated ATE on ARIA-H should be
approximately zero.

If it is materially different from zero, that quantifies the residual bias
in the pipeline and can be used to calibrate the transportability estimates.

This module is a thin wrapper around ``causal.runner.main`` — we just
supply a dedicated config that targets the A4 trial arms.
"""
from __future__ import annotations

from pathlib import Path

from causal.runner import main as causal_main


def main(config_path: str | Path) -> int:
    return causal_main(config_path)
