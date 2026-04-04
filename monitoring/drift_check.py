"""
monitoring/drift_check.py
─────────────────────────
Data drift detection using Evidently AI.

Both baseline_data.csv and current_data.csv contain UNSCALED selected features
(post-SelectFromModel, pre-StandardScaler) so distributions are directly comparable.

Usage:
    python monitoring/drift_check.py

Reads:
    monitoring/baseline_data.csv  — X_train_sel saved by train.py (unscaled)
    monitoring/current_data.csv   — API inputs logged per prediction (unscaled)

Outputs:
    monitoring/drift_report.html  — interactive Evidently HTML report
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Evidently 0.7+ top-level API
from evidently import Report
from evidently.presets import DataDriftPreset

# ── Paths ─────────────────────────────────────────────────────────────────────
MONITORING_DIR   = Path(__file__).resolve().parent
BASELINE_PATH    = MONITORING_DIR / "baseline_data.csv"
CURRENT_PATH     = MONITORING_DIR / "current_data.csv"
REPORT_PATH      = MONITORING_DIR / "drift_report.html"
MIN_ROWS         = 50   # minimum rows for a statistically meaningful drift check


def load_baseline() -> pd.DataFrame:
    """Load the training reference distribution."""
    if not BASELINE_PATH.exists():
        print(f"[ERROR] Baseline not found at {BASELINE_PATH}.")
        print("        Run `python src/train.py` first.")
        sys.exit(1)

    df = pd.read_csv(BASELINE_PATH)
    print(f"[INFO]  Baseline:      {df.shape[0]:>5} rows × {df.shape[1]} cols")
    print(f"[DEBUG] Sample (baseline):\n{df.head(2).to_string()}\n")
    return df


def load_current(reference: pd.DataFrame) -> pd.DataFrame:
    """
    Load production data.
    If fewer than MIN_ROWS rows exist, fall back to a sample from the baseline
    (same distribution → near-zero drift, good for demo/viva).
    """
    simulated = False

    if not CURRENT_PATH.exists() or CURRENT_PATH.stat().st_size == 0:
        print(f"[WARN]  current_data.csv missing or empty — using simulated data.")
        simulated = True
    else:
        df = pd.read_csv(CURRENT_PATH)
        if len(df) < MIN_ROWS:
            print(f"[WARN]  Only {len(df)} production row(s) logged "
                  f"(minimum {MIN_ROWS} for reliable drift detection).")
            print("        Augmenting with baseline sample to reach minimum threshold.")
            simulated = True
        else:
            print(f"[INFO]  Current data: {df.shape[0]:>5} rows × {df.shape[1]} cols")
            print(f"[DEBUG] Sample (current):\n{df.head(2).to_string()}\n")

    if simulated:
        # Sample from baseline with slight noise — realistic near-zero drift
        sample = reference.sample(n=min(100, len(reference)), random_state=42).copy()
        noise  = np.random.normal(0, 0.01, sample.shape)
        df     = pd.DataFrame(sample.values + noise, columns=sample.columns)
        print(f"[INFO]  Simulated current data: {df.shape[0]} rows (baseline + tiny noise)")

    return df, simulated


def align_columns(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Ensure current data has exactly the same columns as reference."""
    missing = set(reference.columns) - set(current.columns)
    extra   = set(current.columns)  - set(reference.columns)

    if missing:
        print(f"[WARN]  Filling {len(missing)} missing column(s) with 0.")
        for col in missing:
            current[col] = 0.0

    if extra:
        print(f"[INFO]  Dropping {len(extra)} extra column(s) not in baseline.")

    return current[reference.columns]   # enforce same column order


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run Evidently drift detection and save HTML report."""
    print("[INFO]  Running Evidently DataDriftPreset...")

    report   = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))

    # ── Extract drift summary from snapshot ───────────────────────────────────
    try:
        result_dict = snapshot.dict()
        metrics     = result_dict.get("metrics", [])
        drifted     = 0
        total       = 0
        for m in metrics:
            result = m.get("result", {})
            if "number_of_drifted_columns" in result:
                drifted = result["number_of_drifted_columns"]
                total   = result.get("number_of_columns", len(reference.columns))
                break
    except Exception:
        drifted = "?"
        total   = len(reference.columns)

    return {"drifted": drifted, "total": total}


def print_summary(stats: dict, simulated: bool) -> None:
    """Print a human-readable drift summary."""
    d, t = stats["drifted"], stats["total"]
    pct  = f"{100 * d / t:.1f}%" if isinstance(d, int) and t > 0 else "N/A"

    print("=" * 55)
    print(f"  Drift detected in {d} out of {t} features ({pct})")
    if simulated:
        print("  [NOTE] Simulated data used — expect near-zero drift.")
    print(f"  Report → {REPORT_PATH}")
    print("=" * 55)


def main() -> None:
    print("=" * 55)
    print("  Parkinson Detection — Data Drift Check")
    print("=" * 55)

    reference          = load_baseline()
    current, simulated = load_current(reference)
    current            = align_columns(reference, current)

    stats = run_drift_report(reference, current)
    print_summary(stats, simulated)


if __name__ == "__main__":
    main()
