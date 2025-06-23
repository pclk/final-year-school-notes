import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

# Load the processed data
data = np.load("processed_data.npz", allow_pickle=True)

# Load arrays
X = data["X"]
y = data["y"]
feature_names = data["feature_names"]

# Convert to DataFrame for easier analysis
df = pd.DataFrame(X, columns=feature_names)
df["salary"] = y

# Create Profile Report
profile = ProfileReport(df, title="Salary Prediction Dataset Profiling Report")

# Save the report to HTML
profile.to_file("salary_data_profile_report.html")

print("Profile report has been generated as 'salary_data_profile_report.html'")
