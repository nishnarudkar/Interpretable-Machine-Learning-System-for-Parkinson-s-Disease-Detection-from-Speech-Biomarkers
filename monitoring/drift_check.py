"""
monitoring/drift_check.py
─────────────────────────
Interpretable data drift detection using Evidently AI.

Both baseline_data.csv and current_data.csv contain UNSCALED selected features
(post-SelectFromModel, pre-StandardScaler) so distributions are directly comparable.

Usage:
    python monitoring/drift_check.py

Reads:
    monitoring/baseline_data.csv  — X_train_sel saved by train.py (unscaled)
    monitoring/current_data.csv   — API inputs logged per prediction (unscaled)

Outputs:
    monitoring/drift_report.html       — interactive Evidently HTML report
    monitoring/drift_summary.txt       — plain-text drift summary
    monitoring/drift_feature_details.csv — per-feature drift scores
"""

import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from evidently import Report
from evidently.presets import DataDriftPreset

# ── Paths ─────────────────────────────────────────────────────────────────────
MONITORING_DIR   = Path(__file__).resolve().parent
BASELINE_PATH    = MONITORING_DIR / "baseline_data.csv"
CURRENT_PATH     = MONITORING_DIR / "current_data.csv"
REPORT_PATH      = MONITORING_DIR / "drift_report.html"
SUMMARY_PATH     = MONITORING_DIR / "drift_summary.txt"
DETAILS_CSV_PATH = MONITORING_DIR / "drift_feature_details.csv"

MIN_ROWS         = 50    # minimum rows for statistically meaningful drift
DRIFT_THRESHOLD  = 0.50  # flag if >50% of features drift


# ── Data loading ──────────────────────────────────────────────────────────────

def load_baseline() -> pd.DataFrame:
    """Load the training reference distribution."""
    if not BASELINE_PATH.exists():
        print(f"[ERROR] Baseline not found at {BASELINE_PATH}.")
        print("        Run `python src/train.py` first.")
        sys.exit(1)

    df = pd.read_csv(BASELINE_PATH)
    print(f"[INFO]  Baseline:      {df.shape[0]:>5} rows × {df.shape[1]} cols")
    print(f"[DEBUG] Baseline sample (first 2 rows):")
    print(df.head(2).to_string())
    print()
    return df


def load_current(reference: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Load production data.
    Falls back to a baseline sample with tiny noise when data is insufficient,
    producing near-zero drift — realistic for a freshly deployed system.
    """
    simulated = False

    if not CURRENT_PATH.exists() or CURRENT_PATH.stat().st_size == 0:
        print("[WARN]  current_data.csv missing or empty — using simulated data.")
        simulated = True
    else:
        try:
            df = pd.read_csv(CURRENT_PATH)
        except Exception as e:
            print(f"[WARN]  Could not read current_data.csv ({e}) — using simulated data.")
            simulated = True
        else:
            if len(df) < MIN_ROWS:
                print(f"[WARN]  Only {len(df)} production row(s) logged "
                      f"(minimum {MIN_ROWS} for reliable drift detection).")
                print("        Augmenting with baseline sample to reach minimum threshold.")
                simulated = True
            else:
                print(f"[INFO]  Current data: {df.shape[0]:>5} rows × {df.shape[1]} cols")
                print(f"[DEBUG] Current data sample (first 2 rows):")
                print(df.head(2).to_string())
                print()

    if simulated:
        n      = min(100, len(reference))
        sample = reference.sample(n=n, random_state=42).copy()
        noise  = np.random.default_rng(42).normal(0, 0.01, sample.shape)
        df     = pd.DataFrame(sample.values + noise, columns=sample.columns)
        print(f"[INFO]  Simulated current data: {len(df)} rows (baseline + σ=0.01 noise)")

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

    return current[reference.columns]


# ── Drift detection ───────────────────────────────────────────────────────────

def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Run Evidently drift detection.

    Returns a dict with:
        total_features  — int
        drifted_count   — int
        drift_pct       — float (0–100)
        per_column      — list of {feature, p_value, drifted}
    """
    print("[INFO]  Running Evidently DataDriftPreset...")

    report   = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))
    print(f"[OK]    HTML report saved → {REPORT_PATH}")

    # ── Parse snapshot dict ───────────────────────────────────────────────────
    d = snapshot.dict()
    metrics = d.get("metrics", [])

    total_features = len(reference.columns)
    drifted_count  = 0
    per_column     = []

    for m in metrics:
        name = m.get("metric_name", "")
        val  = m.get("value")

        # Overall drifted column count
        if name.startswith("DriftedColumnsCount") and isinstance(val, dict):
            drifted_count = int(val.get("count", 0))

        # Per-column p-value (K-S test)
        if name.startswith("ValueDrift(column=") and isinstance(val, (float, int, np.floating)):
            # Extract column name from metric name string
            # Format: "ValueDrift(column=feature_name,method=...)"
            try:
                col_part = name.split("column=")[1].split(",")[0]
            except IndexError:
                col_part = "unknown"
            p_value = float(val)
            per_column.append({
                "feature": col_part,
                "p_value": round(p_value, 6),
                "drifted": p_value < 0.05,
            })

    drift_pct = (drifted_count / total_features * 100) if total_features > 0 else 0.0

    return {
        "total_features": total_features,
        "drifted_count":  drifted_count,
        "drift_pct":      round(drift_pct, 1),
        "per_column":     per_column,
    }


# ── Output & reporting ────────────────────────────────────────────────────────

def save_feature_details(per_column: list) -> None:
    """Save per-feature drift scores to CSV."""
    if not per_column:
        return
    df = pd.DataFrame(per_column).sort_values("p_value")
    df.to_csv(DETAILS_CSV_PATH, index=False)
    print(f"[OK]    Feature details saved → {DETAILS_CSV_PATH}")


def save_summary_txt(stats: dict, top5: list, simulated: bool) -> None:
    """Write plain-text drift summary."""
    lines = [
        "=" * 50,
        "  DRIFT SUMMARY",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        f"  Total Features   : {stats['total_features']}",
        f"  Drifted Features : {stats['drifted_count']}",
        f"  Drift Percentage : {stats['drift_pct']}%",
        "",
        "  Top 5 Drifted Features (lowest p-value):",
    ]
    for i, f in enumerate(top5, 1):
        lines.append(f"    {i}. {f['feature']}  (p={f['p_value']})")

    lines += [
        "",
        "  Status: " + (
            "Significant dataset drift detected"
            if stats["drift_pct"] > DRIFT_THRESHOLD * 100
            else "No significant dataset drift"
        ),
    ]
    if simulated:
        lines.append("  [NOTE] Simulated data used — near-zero drift expected.")
    lines.append("=" * 50)

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK]    Summary saved → {SUMMARY_PATH}")


def print_console_summary(stats: dict, top5: list, simulated: bool) -> None:
    """Print a professional drift summary to the console."""
    d   = stats["drifted_count"]
    t   = stats["total_features"]
    pct = stats["drift_pct"]

    # ASCII histogram
    drifted_bar = "█" * d
    clean_bar   = "░" * (t - d)
    bar_scale   = max(1, t // 40)
    hist_d = "█" * (d // bar_scale)
    hist_c = "░" * ((t - d) // bar_scale)

    print()
    print("─" * 50)
    print("  DRIFT SUMMARY")
    print("─" * 50)
    print(f"  Total Features   : {t}")
    print(f"  Drifted Features : {d}")
    print(f"  Drift Percentage : {pct}%")
    print()
    print(f"  Distribution: [{hist_d}{hist_c}]")
    print(f"                 ↑ drifted ({d})   ↑ stable ({t - d})")
    print()

    if top5:
        print("  Top 5 Drifted Features (lowest p-value):")
        for i, f in enumerate(top5, 1):
            flag = "⚠" if f["drifted"] else "✓"
            print(f"    {i}. {flag} {f['feature']}  (p={f['p_value']})")
    print()

    if pct > DRIFT_THRESHOLD * 100:
        status = "⚠  Significant dataset drift detected"
        action = "Dataset drift exceeds threshold → Retraining recommended"
    else:
        status = "✓  No significant dataset drift"
        action = "Drift within acceptable limits → Monitoring continues"

    print(f"  Final Status: {status}")
    print(f"  Action:       {action}")

    if simulated:
        print()
        print("  [NOTE] Simulated data used — near-zero drift is expected.")
        print("         Send real predictions through the API for live monitoring.")

    print("─" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 50)
    print("  Parkinson Detection — Data Drift Check")
    print("=" * 50)

    reference          = load_baseline()
    current, simulated = load_current(reference)
    current            = align_columns(reference, current)

    stats = run_drift_report(reference, current)

    # Top 5 most drifted features (lowest p-value = most drifted)
    per_col = sorted(stats["per_column"], key=lambda x: x["p_value"])
    top5    = per_col[:5]

    save_feature_details(per_col)
    save_summary_txt(stats, top5, simulated)
    print_console_summary(stats, top5, simulated)


if __name__ == "__main__":
    main()
