import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv("cleaned_jobs.csv")

# Convert categorical columns to numeric for correlation analysis
df_numeric = df.copy()

# Convert categorical columns using label encoding
categorical_columns = [
    "country",
    "location_flexibility",
    "contract_type",
    "education_level",
    "seniority",
]
for col in categorical_columns:
    df_numeric[col] = pd.Categorical(df_numeric[col]).codes

# Select relevant columns for correlation
columns_for_correlation = [
    "yearly_salary_midpoint",
    "min_years_experience",
    "country",
    "location_flexibility",
    "contract_type",
    "education_level",
    "seniority",
]

# Create correlation matrix
correlation_matrix = df_numeric[columns_for_correlation].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True
)
plt.title("Correlation Matrix of Job Features")
plt.tight_layout()
plt.show()

# Print key correlations with salary
salary_correlations = correlation_matrix["yearly_salary_midpoint"].sort_values(
    ascending=False
)
print("\nCorrelations with yearly_salary_midpoint:")
print(salary_correlations)

print("\nSuggested features for training based on correlation analysis:")
print("1. Primary features (stronger correlations):")
for feature, corr in salary_correlations.items():
    if abs(corr) >= 0.1 and feature != "yearly_salary_midpoint":
        print(f"   - {feature}: {corr:.3f}")

print("\n2. Secondary features (weaker correlations):")
for feature, corr in salary_correlations.items():
    if abs(corr) < 0.1 and feature != "yearly_salary_midpoint":
        print(f"   - {feature}: {corr:.3f}")

# Basic statistics for numerical features
print("\nBasic statistics for salary:")
print(df["yearly_salary_midpoint"].describe())

# Distribution of categorical variables
print("\nValue counts for categorical variables:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head())
