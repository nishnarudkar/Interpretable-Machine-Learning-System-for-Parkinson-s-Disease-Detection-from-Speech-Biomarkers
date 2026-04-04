import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("Loading data...")

# Load baseline data (training reference)
reference_data = pd.read_csv("monitoring/baseline_data.csv")

# Load current production data
current_data = pd.read_csv("monitoring/current_data.csv")

# Ensure same columns
current_data = current_data[reference_data.columns]

print("Running drift detection...")

# Create Evidently report
report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save report
report.save_html("monitoring/drift_report.html")

print("✅ Drift report generated at monitoring/drift_report.html")