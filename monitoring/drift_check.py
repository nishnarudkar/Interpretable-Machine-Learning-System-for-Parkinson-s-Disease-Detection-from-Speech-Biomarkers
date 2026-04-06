"""
monitoring/drift_check.py
─────────────────────────
Interpretable data drift detection using the Kolmogorov-Smirnov (KS) test
with feature-importance-aware retraining recommendations.

Both baseline_data.csv and current_data.csv contain UNSCALED selected features
(post-SelectFromModel, pre-StandardScaler) so distributions are directly comparable.

Usage:
    python monitoring/drift_check.py

Reads:
    monitoring/baseline_data.csv       — X_train_sel saved by train.py (unscaled)
    monitoring/current_data.csv        — API inputs logged per prediction (unscaled)
    artifacts/feature_config.json      — top important features from XGBoost

Outputs:
    monitoring/drift_report.html       — interactive Evidently HTML report
    monitoring/drift_summary.txt       — plain-text drift summary
    monitoring/drift_feature_details.csv — per-feature KS test results
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Paths ─────────────────────────────────────────────────────────────────────
MONITORING_DIR   = Path(__file__).resolve().parent
ROOT_DIR         = MONITORING_DIR.parent
BASELINE_PATH    = MONITORING_DIR / "baseline_data.csv"
CURRENT_PATH     = MONITORING_DIR / "current_data.csv"
REPORT_PATH      = MONITORING_DIR / "drift_report.html"
SUMMARY_PATH     = MONITORING_DIR / "drift_summary.txt"
DETAILS_CSV_PATH = MONITORING_DIR / "drift_feature_details.csv"
FEATURE_CONFIG   = ROOT_DIR / "artifacts" / "feature_config.json"

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_ROWS        = 50     # minimum rows for statistically meaningful drift
P_VALUE_THRESH  = 0.05   # KS test significance threshold

# Drift severity bands (% of features drifted)
SEVERITY_LOW      = 20   # 0–20%   → No / Low drift
SEVERITY_MODERATE = 50   # 20–50%  → Moderate drift
                         # >50%    → High drift

# Retraining triggers
RETRAIN_HIGH_DRIFT        = 50   # % — always retrain above this
RETRAIN_MODERATE_DRIFT    = 30   # % — retrain if important features also drift


# ── 1. Data loading ───────────────────────────────────────────────────────────

def load_baseline() -> pd.DataFrame:
    """Load the training reference distribution."""
    if not BASELINE_PATH.exists():
        print(f"[ERROR] Baseline not found at {BASELINE_PATH}.")
        print("        Run `python src/train.py` first.")
        sys.exit(1)
    df = pd.read_csv(BASELINE_PATH)
    print(f"[INFO]  Baseline:     {df.shape[0]:>5} rows × {df.shape[1]} cols")
    return df


def load_current(reference: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Load production data.
    Falls back to a baseline sample with tiny noise when data is insufficient.
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


def load_important_features() -> list[str]:
    """Load top important features from XGBoost feature config."""
    if not FEATURE_CONFIG.exists():
        return []
    try:
        with open(FEATURE_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("top_features", [])
    except Exception:
        return []


# ── 2. KS test computation ────────────────────────────────────────────────────

def compute_ks_drift(reference: pd.DataFrame,
                     current: pd.DataFrame) -> list[dict]:
    """
    Run two-sample KS test for each feature.

    Returns list of dicts sorted by p_value ascending (most drifted first):
        feature      — feature name
        ks_stat      — KS statistic (0–1, higher = more different)
        p_value      — raw p-value
        p_display    — human-readable p-value string
        drifted      — bool (p_value < P_VALUE_THRESH)
    """
    results = []
    for col in reference.columns:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        if len(ref_vals) < 2 or len(cur_vals) < 2:
            continue

        ks_stat, p_value = scipy_stats.ks_2samp(ref_vals, cur_vals)

        # Format p-value: avoid misleading "0.0"
        if p_value == 0.0 or p_value < 1e-6:
            p_display = "< 1e-6"
        elif p_value < 0.001:
            p_display = f"{p_value:.2e}"
        else:
            p_display = f"{p_value:.4f}"

        results.append({
            "feature":   col,
            "ks_stat":   round(float(ks_stat), 6),
            "p_value":   float(p_value),
            "p_display": p_display,
            "drifted":   bool(p_value < P_VALUE_THRESH),
        })

    return sorted(results, key=lambda x: x["p_value"])


# ── 3. Drift classification ───────────────────────────────────────────────────

def classify_severity(drift_pct: float) -> str:
    """Classify drift severity based on percentage of drifted features."""
    if drift_pct <= SEVERITY_LOW:
        return "Low"
    elif drift_pct <= SEVERITY_MODERATE:
        return "Moderate"
    else:
        return "High"


def decide_retraining(drift_pct: float,
                      per_column: list[dict],
                      important_features: list[str]) -> tuple[bool, str]:
    """
    Intelligent retraining decision based on three triggers:
      1. Drift > 50%  (high drift — always retrain)
      2. Drift > 30% AND important features are affected
      3. (Future) Model performance degradation

    Returns (should_retrain: bool, reason: str)
    """
    drifted_features = {r["feature"] for r in per_column if r["drifted"]}

    # Trigger 1 — high overall drift
    if drift_pct > RETRAIN_HIGH_DRIFT:
        return True, f"High drift ({drift_pct:.1f}% > {RETRAIN_HIGH_DRIFT}% threshold)"

    # Trigger 2 — moderate drift + important features affected
    if drift_pct > RETRAIN_MODERATE_DRIFT and important_features:
        affected_important = [f for f in important_features if f in drifted_features]
        if affected_important:
            return True, (
                f"Moderate drift ({drift_pct:.1f}%) with {len(affected_important)} "
                f"important feature(s) drifted: {', '.join(affected_important)}"
            )

    return False, "Drift within acceptable limits"


# ── 4. Evidently HTML report ──────────────────────────────────────────────────

def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Generate interactive Evidently HTML report (optional, non-fatal)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report   = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(current_data=current, reference_data=reference)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(REPORT_PATH))
        print(f"[OK]    Evidently HTML report saved → {REPORT_PATH}")
    except Exception as e:
        print(f"[WARN]  Evidently report skipped: {e}")


# ── 5. Reporting ──────────────────────────────────────────────────────────────

def save_feature_details(per_column: list[dict]) -> None:
    """Save per-feature KS test results to CSV."""
    if not per_column:
        return
    df = pd.DataFrame([{
        "feature":   r["feature"],
        "ks_stat":   r["ks_stat"],
        "p_value":   r["p_value"],
        "p_display": r["p_display"],
        "drifted":   r["drifted"],
    } for r in per_column])
    df.to_csv(DETAILS_CSV_PATH, index=False)
    print(f"[OK]    Feature details saved → {DETAILS_CSV_PATH}")


def save_summary_txt(stats: dict, simulated: bool) -> None:
    """Write plain-text drift summary."""
    lines = [
        "=" * 55,
        "  DRIFT SUMMARY",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 55,
        f"  Total Features          : {stats['total_features']}",
        f"  Drifted Features        : {stats['drifted_count']}",
        f"  Drift Percentage        : {stats['drift_pct']:.1f}%",
        f"  Drift Severity          : {stats['severity']}",
        f"  Retraining Recommended  : {'Yes' if stats['retrain'] else 'No'}",
        f"  Retraining Reason       : {stats['retrain_reason']}",
        "",
        "  Top 5 Drifted Features (lowest p-value):",
    ]
    for i, f in enumerate(stats["top5"], 1):
        imp = " [IMPORTANT]" if f.get("important") else ""
        lines.append(f"    {i}. {f['feature']}  (p={f['p_display']}, KS={f['ks_stat']:.4f}){imp}")

    if stats["important_drifted"]:
        lines += [
            "",
            f"  Important Features Drifted ({len(stats['important_drifted'])}):",
        ]
        for f in stats["important_drifted"]:
            lines.append(f"    - {f}")

    if simulated:
        lines += [
            "",
            "  [NOTE] Simulated data used — near-zero drift expected.",
            "         Send real predictions through the API for live monitoring.",
        ]
    lines.append("=" * 55)

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK]    Summary saved → {SUMMARY_PATH}")


def print_console_summary(stats: dict, simulated: bool) -> None:
    """Print structured drift summary to console."""
    d   = stats["drifted_count"]
    t   = stats["total_features"]
    pct = stats["drift_pct"]

    bar_scale = max(1, t // 40)
    hist_d    = "█" * (d // bar_scale)
    hist_c    = "░" * ((t - d) // bar_scale)

    severity_color = {
        "Low":      "✓",
        "Moderate": "⚡",
        "High":     "⚠",
    }

    print()
    print("─" * 55)
    print("  DRIFT SUMMARY")
    print("─" * 55)
    print(f"  Total Features          : {t}")
    print(f"  Drifted Features        : {d}")
    print(f"  Drift Percentage        : {pct:.1f}%")
    print(f"  Drift Severity          : {severity_color.get(stats['severity'], '')} {stats['severity']}")
    print(f"  Retraining Recommended  : {'Yes' if stats['retrain'] else 'No'}")
    print(f"  Reason                  : {stats['retrain_reason']}")
    print()
    print(f"  Distribution: [{hist_d}{hist_c}]")
    print(f"                 ↑ drifted ({d})   ↑ stable ({t - d})")
    print()

    if stats["top5"]:
        print("  Top 5 Drifted Features (KS test, lowest p-value):")
        for i, f in enumerate(stats["top5"], 1):
            imp  = " ★ IMPORTANT" if f.get("important") else ""
            flag = "⚠" if f["drifted"] else "✓"
            print(f"    {i}. {flag} {f['feature']}")
            print(f"         KS stat: {f['ks_stat']:.4f}  |  p-value: {f['p_display']}{imp}")

    if stats["important_drifted"]:
        print()
        print(f"  ★ Important Features Drifted ({len(stats['important_drifted'])}):")
        for f in stats["important_drifted"]:
            print(f"    - {f}")

    if simulated:
        print()
        print("  [NOTE] Simulated data used — near-zero drift is expected.")
        print("         Send real predictions through the API for live monitoring.")

    print("─" * 55)


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  Parkinson Detection — Data Drift Check")
    print("=" * 55)

    reference           = load_baseline()
    current, simulated  = load_current(reference)
    current             = align_columns(reference, current)
    important_features  = load_important_features()

    if important_features:
        print(f"[INFO]  Important features loaded: {important_features}")

    # KS test
    per_column = compute_ks_drift(reference, current)

    drifted_count = sum(1 for r in per_column if r["drifted"])
    total         = len(per_column)
    drift_pct     = (drifted_count / total * 100) if total > 0 else 0.0
    severity      = classify_severity(drift_pct)
    retrain, reason = decide_retraining(drift_pct, per_column, important_features)

    # Tag important features in results
    imp_set = set(important_features)
    for r in per_column:
        r["important"] = r["feature"] in imp_set

    important_drifted = [
        r["feature"] for r in per_column
        if r["drifted"] and r["feature"] in imp_set
    ]

    top5 = per_column[:5]

    stats = {
        "total_features":    total,
        "drifted_count":     drifted_count,
        "drift_pct":         drift_pct,
        "severity":          severity,
        "retrain":           retrain,
        "retrain_reason":    reason,
        "top5":              top5,
        "important_drifted": important_drifted,
        "generated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Outputs
    save_feature_details(per_column)
    save_summary_txt(stats, simulated)
    run_evidently_report(reference, current)
    print_console_summary(stats, simulated)


if __name__ == "__main__":
    main()
