import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference and current data
reference = pd.read_csv("data/house_prices.csv")
current = pd.read_csv("data/house_prices.csv")  # In real scenario, this would be new incoming data

# Create a data drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

# Save report as HTML
report.save_html("reports/data_drift_report.html")
print("✅ Report generated: reports/data_drift_report.html")
