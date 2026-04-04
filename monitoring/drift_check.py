"""
monitoring/drift_check.py
─────────────────────────
Data drift detection using Evidently.

Usage:
    python monitoring/drift_check.py

Reads:
    monitoring/baseline_data.csv  — reference distribution (X_train from training)
    monitoring/current_data.csv   — production inputs logged by the API

Outputs:
    monitoring/drift_report.html  — interactive HTML drift report
"""

import sys
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ── Paths ─────────────────────────────────────────────────────────────────────
MONITORING_DIR   = Path(__file__).resolve().parent
BASELINE_PATH    = MONITORING_DIR / "baseline_data.csv"
CURRENT_PATH     = MONITORING_DIR / "current_data.csv"
REPORT_PATH      = MONITORING_DIR / "drift_report.html"
MIN_CURRENT_ROWS = 10   # need at least this many rows to run a meaningful report


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline and current data, aligning columns."""

    # ── Baseline ──────────────────────────────────────────────────────────────
    if not BASELINE_PATH.exists():
        print(f"[ERROR] Baseline data not found at {BASELINE_PATH}.")
        print("        Run `python src/train.py` first to generate it.")
        sys.exit(1)

    reference = pd.read_csv(BASELINE_PATH)
    print(f"[INFO]  Baseline loaded: {reference.shape[0]} rows × {reference.shape[1]} cols")

    # ── Current ───────────────────────────────────────────────────────────────
    if not CURRENT_PATH.exists():
        print(f"[WARN]  No current data found at {CURRENT_PATH}.")
        print("        Send some predictions through the API first, then re-run.")
        sys.exit(0)

    current = pd.read_csv(CURRENT_PATH)

    if current.empty or len(current) < MIN_CURRENT_ROWS:
        print(f"[WARN]  current_data.csv has only {len(current)} row(s) "
              f"(minimum {MIN_CURRENT_ROWS} required for drift analysis).")
        print("        Send more predictions through the API, then re-run.")
        sys.exit(0)

    print(f"[INFO]  Current data loaded: {current.shape[0]} rows × {current.shape[1]} cols")

    # ── Align columns (handle any mismatch gracefully) ────────────────────────
    common_cols = [c for c in reference.columns if c in current.columns]
    missing_in_current = set(reference.columns) - set(current.columns)

    if missing_in_current:
        print(f"[WARN]  {len(missing_in_current)} column(s) missing from current data — "
              f"filling with 0: {list(missing_in_current)[:5]}...")
        for col in missing_in_current:
            current[col] = 0.0

    reference = reference[reference.columns]
    current   = current[reference.columns]   # enforce same column order

    return reference, current


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Generate Evidently drift report and save as HTML."""

    print("[INFO]  Running Evidently DataDriftPreset...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    print(f"[OK]    Drift report saved → {REPORT_PATH}")


def main() -> None:
    print("=" * 55)
    print("  Parkinson Detection — Data Drift Check")
    print("=" * 55)

    reference, current = load_data()
    run_drift_report(reference, current)

    print("=" * 55)
    print("  Done. Open drift_report.html to review results.")
    print("=" * 55)


if __name__ == "__main__":
    main()
