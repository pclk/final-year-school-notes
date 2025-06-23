r"""°°°
# Data Understanding and Preparation
°°°"""
r"""°°°
Our goal is to ultimately develop a robust machine learning model that accurately predicts insurance premium amount ("Premium Amount")
°°°"""
r"""°°°
## Data Understanding
°°°"""

# from google.colab import drive
# drive.mount('/content/drive')


import pandas as pd


# change to the appropriate filepath to the csv
GOOGLE_COLAB_FILE_DIR="/content/drive/MyDrive/school/AML/insurance_prediction.csv"

df = pd.read_csv("insurance_prediction.csv")


df.info()



# Calculate missing values
total_rows = len(df)
missing_stats = df.isnull().sum()

# Convert to percentage and filter columns with missing values
missing_stats = missing_stats[missing_stats > 0]
missing_pct = (missing_stats / total_rows * 100).round(1)

# Create a DataFrame with both counts and percentages
missing_df = pd.DataFrame(
    {"Missing Count": missing_stats, "Missing Percentage": missing_pct}
)

# Sort by missing percentage in descending order
missing_df = missing_df.sort_values("Missing Percentage", ascending=False)

# Format the output
print("Missing Value Analysis:")
print("-" * 60)
for idx, row in missing_df.iterrows():
    print(
        f"{idx:<20} has {int(row['Missing Count']):>6} ({row['Missing Percentage']:>4.1f}%) missing values"
    )

r"""°°°
There are large gaps in important information like Occupation, Credit Score and other factors.

Reasoning for Missing values are rarely random. For example, individuals from lower socioeconomic backgrounds might be less likely to provide detailed financial information, leading to higher proportion for missing values.

If these missing values are not addressed correctly, especially due to the large amount of missing values, this can lead to biased models that unfairly disadvantage certain groups. For example, An AI model trained on data with systematic missingness might learn "missing values" = "high risk".

One popular way to deal with missing values is imputation, which is the process of filling missing values. However, this can introduce its own biases. Simple methods like mean or median can distort the distribution of data. Imagine if Previous Claims' missing values was replaced with mean. The distribution would look like a normal distribution even though it, in reality, might not. More sophisticated methods like regression imputation or k-nearest neighbors can introduce complex dependencies that might not reflect reality.

In reality, the data scientist facing such an issue should communicate back with the data engineer and subject matter experts regarding the reasoning for missing values, and gain additional insights from there, which can potentially allow the data scientist to create interaction terms based on these insights for the model to learn.

After the data scientist create interaction terms, this step must be shared to increase the transparency of the model cleaning process. The company should have internal policies to document and store information in regards to model. Many insurance companies that treat their AI models as "black boxes" make it difficult to understand how decisions are made, eroding trust and making it challenging to identify biases.

For the purposes of this assignment, we assume that the data scientist was able to gain the insights to decide that **mean imputation** is the correct and fair way to address the issue.
°°°"""

# Calculate zero values
zero_stats = (df == 0).sum()
zero_pct = (zero_stats / total_rows * 100).round(1)

# Filter only numeric columns with zeros and create DataFrame
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
zero_df = pd.DataFrame(
    {"Zero Count": zero_stats[numeric_cols], "Zero Percentage": zero_pct[numeric_cols]}
)
zero_df = zero_df[zero_df["Zero Count"] > 0].sort_values(
    "Zero Percentage", ascending=False
)

# Format the output for zero values
print("\nZero Value Analysis:")
print("-" * 60)
for idx, row in zero_df.iterrows():
    print(
        f"{idx:<20} has {int(row['Zero Count']):>6} ({row['Zero Percentage']:>4.1f}%) zeros"
    )

r"""°°°
While missing values signify non-response, system errors, or data entry omissions, Zero values represent explicit data points. It could mean that a customer had 0 previous claims, or a placeholder for "not applicable", which may explain the large amount of 0 previous claims.

Therefore, zero values have a high chance of being misinterpreted or mishandled by the various processes that the raw data goes through to get to the analysis stage. Zero values may be created from several reasons, and that needs to be checked by the data scientist, otherwise several ethical issues, like inaccurate predictions or unfair outcomes may occur, introducing bias and obscuring important information.

For the purposes of this assignment, we assume that the data scientist was able to gain the insights to check that all the zeros mean that the items are zero, for example, 0 previous claims, 0 dependents, vehicle is brand new.

This means that no actions are required to address the zeros.
°°°"""

# Specify the columns we want to analyze
categorical_cols = [
    "Gender",
    "Marital Status",
    "Number of Dependents",
    "Education Level",
    "Occupation",
    "Location",
    "Policy Type",
    "Customer Feedback",
    "Smoking Status",
    "Exercise Frequency",
    "Property Type",
]

# Display distribution for each specified variable
print("\nCategorical Variables Distribution:")
print("-" * 80)
for col in categorical_cols:
    value_counts = df[col].value_counts()
    percentages = (value_counts / len(df) * 100).round(1)

    # Sort by percentage in descending order
    percentages = percentages.sort_values(ascending=False)
    value_counts = value_counts[percentages.index]

    print(f"\n{col}:")
    print("-" * 70)

    # Calculate max value length for alignment
    max_val_length = max(len(str(val)) for val in value_counts.index)

    # Create visual representation
    for val, count in value_counts.items():
        pct = percentages[val]
        bar_length = int(pct / 2)  # Scale bar length (50 = full width)
        bar = "█" * bar_length
        print(f"{str(val):<{max_val_length+2}} : {count:>7} ({pct:>4.1f}%) {bar}")

    # Add distribution indicator
    unique_vals = len(value_counts)
    max_pct = percentages.max()
    min_pct = percentages.min()
    pct_range = max_pct - min_pct

    if unique_vals <= 5 and pct_range < 10:
        print(
            f"{'★ EVEN DISTRIBUTION ★':>{max_val_length+2}} : max-min spread = {pct_range:.1f}%"
        )
    print()


r"""°°°
All categorical variables are extremely even. Statistically speaking, this is improbable in real-world data. This strongly suggest that this is synthetic or artificially balanced data. Synthetic data is useful in educational sense, where students are expected to apply machine learning models to it and return relevant AI ethics info, but in the real world, synthetic data must be labelled clearly, like 'synthetic_insurance_prediction.csv'.

°°°"""

# Create correlation matrix plot
import seaborn as sns
import matplotlib.pyplot as plt

# Select numeric columns only
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_cols].corr()

# Create a larger figure
plt.figure(figsize=(12, 10))

# Create heatmap
sns.heatmap(correlation_matrix,
            annot=True,            # Show correlation values
            cmap='coolwarm',       # Color scheme
            center=0,              # Center the colormap at 0
            fmt='.2f',            # Format correlation values to 2 decimal places
            square=True,           # Make cells square
            linewidths=0.5,        # Add grid lines
            cbar_kws={"shrink": .5}) # Adjust colorbar size

plt.title('Correlation Matrix of Numeric Variables', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

r"""°°°
Signs of poor data:
- Highest correlation is between Credit Score and Annual Income. This indicates possible multicollinearity
- Looking at the rightmost column, we see that Premium Amount has very weak correlation with the rest of variables. the most is "Previous Claims", which is r=0.05. This indicates features that are not important to the target variable of "Premium Amount".

The  signs of poor data are looking at the linear correlation of features and our target variable. While they may not capture non-linear relationships, we are not building a Deep Learning Model that can capture complex relationships. These bad signs mean that models will perform badly.
°°°"""

# Statistical Summary for Numeric Variables
print("\nStatistical Summary of Numeric Variables:")
print("-" * 80)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
stats_df = df[numeric_cols].describe()
print(stats_df.round(2))

# Check for outliers using IQR method
print("\nOutlier Analysis using IQR method:")
print("-" * 80)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    if len(outliers) > 0:
        print(f"\n{col}:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Percentage of outliers: {(len(outliers)/len(df)*100):.2f}%")
        print(f"Range of outliers: {outliers.min():.2f} to {outliers.max():.2f}")


r"""°°°
We have some outliers in Annual Income and Premium Amount. Outliers can either reflect genuine anomalies, or be the result of errors, biases ow unique circumstances. One thing that is certain is that outliers will skew the model's understanding of the underlying relationships between variables.

In the worst case scenario, the outliers are misleading, and it for example, learns the few individuals with high incomes and overestimate the incomes, leading to inaccuracy and unfair pricing. Outliers can also introduce bias. If the outliers are more prevalent in certain demographic groups, the model might unfairly penalize individuals from those groups.

looking at the number, percentage and range of outliers, it seems like these numbers can be rather reasonable.

1. **Annual Income (99,584 to 149,997)**
- The range appears reasonable because:
  - The median income is 23,911
  - High-income individuals earning 100K-150K annually is plausible
  - The upper limit of ~150K suggests the data might be capped, which is a common practice
  - The 5.59% proportion of outliers is within a reasonable range (typically <10% is acceptable)

2. **Previous Claims (6 to 9)**
- This range needs careful consideration:
  - Having 6-9 previous claims is unusually high
  - But Only 0.03% of customers fall in this range (369 cases out of 835,971)
  - These could be:
    - High-risk customers
    - Potential fraud cases

3. **Premium Amount (3,002 to 4,999)**
- This range appears reasonable because:
  - The median premium is 872
  - Higher premiums (3K-5K) could correspond to:
    - High-risk customers
    - Comprehensive coverage
    - Luxury vehicles
    - Multiple vehicles/policies
  - The 4.11% proportion is reasonable
  - The upper limit of 4,999 suggests a policy cap

°°°"""

# Analyze premium distribution for high-claim customers
high_claims_premium = df[df['Previous Claims'] >= 6]['Premium Amount']
print("\nPremium Distribution for High-Claim Customers:")
print(high_claims_premium.describe())

r"""°°°
Previous claims of more than or equal to 6 show very high premium distribution. It is reasonable to think that those with abnormally high previous claims also have abnormally high premium amount. 
°°°"""

# Analyze premium distribution for high-claim customers
high_income_premium = df[df['Annual Income'] >= 99_584]['Premium Amount']
print("\nPremium Distribution for High-Income Customers:")
print(high_income_premium.describe())

r"""°°°
Annual income also show high premium distribution.

Let's take a look at the sorted dataframe from premium amount descending.
°°°"""

df[["Annual Income", "Previous Claims", "Premium Amount"]].sort_values("Premium Amount", ascending=False).head(20)

r"""°°°
It looks like Annual Income and Previous Claims has no real correlation with Premium Amount. This is strange because in real-life, these should be highly correlated.

### Annual Income vs Premium Amount
Typically, the income should correlate because higher income individuals tend to insure more valuable assets, and higher income individuals often opt for more comprehensive coverage.

### Previous Claims vs Premium Amount
Typically, the previous claims should correlate because more claims typically lead to higher premiums, and the industry standard practice is to incerase premiums after claims. https://www.investopedia.com/articles/pf/08/claim-raise-rates.asp#:~:text=Filing%20a%20claim%20often%20results,can%20vary%20widely%20between%20insurers.

This points to data quality issues. Not only is the data synthetic, but also is problematic because it doesn't reflect reality. 
°°°"""

# Check for skewness
print("\nSkewness Analysis:")
print("-" * 80)
skewness = df[numeric_cols].skew()
print("\nSkewness of numeric variables:")
for col, skew in skewness.items():
    print(f"{col:<20}: {skew:>8.2f} {'(Highly Skewed)' if abs(skew) > 1 else '(Moderately Skewed)' if abs(skew) > 0.5 else '(Approximately Symmetric)'}")

r"""°°°
Given the outliers, Annual Income, Previous Claims and Premium Amount are highly left-skewed.

We may use scaling. One approach is to use scaling for SVM and Linear regression related algorithms, but not for tree-based models like random forest, decision tree, XGBoost.

This is because Tree-based models make decisions based of splitting points, not distances or gradients. They ask yes/no questions about features, regardless of their scale. For example, "Is Annual Income <= $45k?".

For linear and SVM, they use mathematical operations or distance calculations, and are scale-dependent.

Therefore, we shall address the left-skewness with np.log1p (logarithm plus 1), which will handle zero values and stretch the concentrated left side of the distribution.

°°°"""
r"""°°°
## Data preparation

°°°"""

# 1. Mean imputation for missing values
for col in df.columns:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)

# 2. One-hot encode categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# 3. Log transform skewed numerical features
skewed_cols = ['Annual Income', 'Previous Claims', 'Premium Amount']
new_skewed_cols = ['Log Annual Income', 'Log Previous Claims', 'Log Premium Amount']
# save it as new_skewed_cols.
for col in skewed_cols:
    # df[col] = np.log1p(df[col])

r"""°°°
## Feature engineering
°°°"""
r"""°°°
We will skip Feature engineering for now since all we want to do is PCA. And PCA should be done after training the Tree-based models for explanability reasons.
°°°"""
r"""°°°
## Model Selection
°°°"""
r"""°°°
### Tree-based models
°°°"""



r"""°°°
### Feature engineering: scaling and dimensoniality reduction for linear and SVM models
°°°"""

# 1. Apply PCA to handle multicollinearity and create new features
from sklearn.decomposition import PCA
# PCA code would go here


r"""°°°
### Linear & SVM models
°°°"""

# Linear & SVM model goes here

r"""°°°
## Performance Measurement
°°°"""



r"""°°°
## Hyperparameter Tuning

We will utilize RandomizedCVSearch to find the best hyperparameters. It is similar to GridSearchCV but it is much quicker and arrives at very similar results.

Here are the hyperparameters we're adjusting, and how it impacts the model performance.
°°°"""



r"""°°°
## Conclusion
°°°"""
