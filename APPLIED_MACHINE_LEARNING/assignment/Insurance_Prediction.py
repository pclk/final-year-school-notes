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
import numpy as np


# change to the appropriate filepath to the csv
GOOGLE_COLAB_FILE_DIR = "/content/drive/MyDrive/school/AML/insurance_prediction.csv"

df = pd.read_csv("insurance_prediction.csv")


df.info()


df.head()


df["Policy Start Date"].head(20)

r"""°°°
We have a date column. Usually, these cannot be directly fed into the model, and additional feature engineering has to be done to the datetime. In this case, we can:

- Extract the day, month, year, dayofweek, quarter of the datetime.
> This can capture cyclical patterns

- Extract the days elapsed since the earliest date.
> This can capture temporal distance information.

This will allow the datetime to transform into a numerical feature that can be used in modelling.
°°°"""

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

For the purposes of this assignment, we assume that the data scientist was able to gain the insights to decide that **mean imputation** for numerical, and **mode imputation** for categorical, is the correct and fair way to address the issue.
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
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
correlation_matrix = df[numeric_cols].corr()

# Create a larger figure
plt.figure(figsize=(12, 10))

# Create heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,  # Show correlation values
    cmap="coolwarm",  # Color scheme
    center=0,  # Center the colormap at 0
    fmt=".2f",  # Format correlation values to 2 decimal places
    square=True,  # Make cells square
    linewidths=0.5,  # Add grid lines
    cbar_kws={"shrink": 0.5},
)  # Adjust colorbar size

plt.title("Correlation Matrix of Numeric Variables", pad=20)
plt.xticks(rotation=45, ha="right")
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
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
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
high_claims_premium = df[df["Previous Claims"] >= 6]["Premium Amount"]
print("\nPremium Distribution for High-Claim Customers:")
print(high_claims_premium.describe())

r"""°°°
Previous claims of more than or equal to 6 show very high premium distribution. It is reasonable to think that those with abnormally high previous claims also have abnormally high premium amount. 
°°°"""

# Analyze premium distribution for high-claim customers
high_income_premium = df[df["Annual Income"] >= 99_584]["Premium Amount"]
print("\nPremium Distribution for High-Income Customers:")
print(high_income_premium.describe())

r"""°°°
Annual income also show high premium distribution.

Let's take a look at the sorted dataframe from premium amount descending.
°°°"""

df[["Annual Income", "Previous Claims", "Premium Amount"]].sort_values(
    "Premium Amount", ascending=False
).head(20)

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
    print(
        f"{col:<20}: {skew:>8.2f} {'(Highly Skewed)' if abs(skew) > 1 else '(Moderately Skewed)' if abs(skew) > 0.5 else '(Approximately Symmetric)'}"
    )

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

# Handle datetime features
datetime_cols = ["Policy Start Date"]
df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"])
for col in datetime_cols:
    # Extract useful datetime components
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    df[f"{col}_quarter"] = df[col].dt.quarter

    # Calculate days since earliest date
    df[f"{col}_days_since_min"] = (df[col] - df[col].min()).dt.days

    # Drop original datetime column
    df = df.drop(columns=[col])
df


df["Number of Dependents"].value_counts()

r"""°°°
We need to be wary of Number of Dependents because it detects the column as a int64 even though its categorical
°°°"""

for col in df.columns:
    if df[col].isnull().any():
        if col == "Number of Dependents":
            # Use mode for 'Number of Dependents'
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
        elif df[col].dtype in ["int64", "float64"]:
            # Use mean for numerical columns
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
        else:
            # Use mode (most frequent value) for categorical columns
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)


df["Number of Dependents"].value_counts()


# One-hot encode categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

r"""°°°
### Why drop_first=True?
The `drop_first=True` argument in `pd.get_dummies` is used to address multicollinearity that arises from one-hot encoding categorical variables, specifically the "dummy variable trap".

Here's how it works and when you should consider using it:

**How `drop_first=True` Impacts Multicollinearity:**

* **Without `drop_first=True`:** When you one-hot encode a categorical variable with *k* categories, `pd.get_dummies` creates *k* binary columns.  For example, if you have a "Color" column with categories "Red", "Blue", and "Green", it will create three columns: "Color_Red", "Color_Blue", and "Color_Green".  In this scenario, if a row is *not* "Red" and *not* "Blue", you can perfectly infer that it *must* be "Green". This means the "Color_Green" column is perfectly predictable from the other two. This perfect predictability leads to **perfect multicollinearity**.

* **With `drop_first=True`:**  `drop_first=True` tells `pd.get_dummies` to drop the *first* category's column. In our "Color" example, if it drops "Color_Red", you'll be left with "Color_Blue" and "Color_Green". Now, if both "Color_Blue" and "Color_Green" are 0, you can still infer that the color is "Red" (the dropped category).  However, you've removed the perfect linear dependency. You have reduced the number of columns to *k-1*, which is sufficient to represent the information without creating perfect multicollinearity.

°°°"""

# Log transform skewed numerical features
skewed_cols = ["Annual Income", "Previous Claims", "Premium Amount"]
log_cols = ["Log Annual Income", "Log Previous Claims", "Log Premium Amount"]

# Create new log-transformed columns while preserving originals
for orig_col, new_col in zip(skewed_cols, log_cols):
    df[new_col] = np.log1p(df[orig_col])

# Print skewness before and after transformation
print("\nSkewness before and after log transformation:")
print("-" * 60)
for orig_col, new_col in zip(skewed_cols, log_cols):
    orig_skew = df[orig_col].skew()
    new_skew = df[new_col].skew()
    print(f"{orig_col:<20}: {orig_skew:>8.2f}")
    print(f"{new_col:<20}: {new_skew:>8.2f}\n")

r"""°°°
log transformation, some of our variables have become negatively skewed, which isn't ideal. Let's try using Box-Cox transformation instead, which can help normalize the data while potentially avoiding negative skewness.
°°°"""

# 3. Transform skewed numerical features using Box-Cox
from scipy import stats

skewed_cols = ["Annual Income", "Previous Claims", "Premium Amount"]
box_cols = [
    "Transformed Annual Income",
    "Transformed Previous Claims",
    "Transformed Premium Amount",
]

# Create new transformed columns while preserving originals
for orig_col, new_col in zip(skewed_cols, box_cols):
    # Add a small constant to handle zeros before Box-Cox
    min_val = df[orig_col].min()
    if min_val <= 0:
        shifted_data = df[orig_col] + abs(min_val) + 1
    else:
        shifted_data = df[orig_col]

    # Apply Box-Cox transformation
    transformed_data, lambda_param = stats.boxcox(shifted_data)
    df[new_col] = transformed_data

# Print skewness before and after transformation
print("\nSkewness before and after Box-Cox transformation:")
print("-" * 60)
for orig_col, new_col in zip(skewed_cols, box_cols):
    orig_skew = df[orig_col].skew()
    new_skew = df[new_col].skew()
    print(f"{orig_col:<20}: {orig_skew:>8.2f}")
    print(f"{new_col:<20}: {new_skew:>8.2f}\n")

r"""°°°
The reason why Box-Cox works, is because it is driven by a formular that finds the optimal λ (lambda) parameter for each variable.
This lamba affects the transformation process of the variables in this way:

- λ = 1 means no transformation needed
- λ = 0.5 means square root transformation
- λ = -1 means reciprocal transformation

Therefore, it mathematically optimizes for normality. That's why we see much better normalization results. It's like choosing the perfect power transformation for each variable's distribution pattern.
°°°"""

# since box-cox returns more normal values, let's drop log columns
for col in log_cols:
    df.drop(col, inplace=True, axis="columns")

r"""°°°
## Feature engineering
°°°"""
r"""°°°
Our feature engineering approach will be split into two phases:

1. First Phase - Tree-based Models:
   - We'll use the original features (including one-hot encoded categoricals)
   - Tree-based models (Random Forest, XGBoost) can:
     - Handle non-linear relationships naturally
     - Work well with raw features
     - Provide feature importance rankings
     - Are interpretable with original features

2. Second Phase - Linear & SVM Models:
   - After tree-based models, we'll apply PCA and Log1p because:
     - Linear/SVM models are sensitive to multicollinearity
     - PCA removes correlations between features
     - Reduces dimensionality, especially important with our many one-hot encoded columns
     - Can improve model performance by removing noise

This two-phase approach maintains interpretability where possible (tree-based) while optimizing performance where needed (linear/SVM).

°°°"""
r"""°°°
## Model Selection & Evaluation
°°°"""
r"""°°°
### Tree-based models

We'll evaluate three tree-based models:
1. Random Forest: Robust against overfitting, handles non-linear relationships
2. XGBoost: Typically highest performance, gradient boosting approach
3. Decision Tree: Baseline model, highly interpretable

Key considerations for our insurance premium prediction:
- High-dimensional data after one-hot encoding
- Potential non-linear relationships between features
- Need for both performance and interpretability
- Presence of outliers in key features
°°°"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import time
from tqdm.auto import tqdm
import lightgbm as lgb

# Prepare features and target
# Exclude log-transformed columns and datetime columns for tree-based models
feature_cols = [
    col
    for col in df.columns
    if col not in ["Premium Amount"] + box_cols and df[col].dtype
]
X = df[feature_cols]
y = df["Premium Amount"]

# Print feature columns being used
print("\nFeatures being used:")
print("-" * 80)
for col in feature_cols:
    print(f"{col}: {df[col].dtype}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize models
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
rf_model = RandomForestRegressor(
    n_estimators=50, random_state=42, max_depth=5, n_jobs=-1
)
xgb_model = xgb.XGBRegressor(
    random_state=42, n_estimators=50, max_depth=5, n_jobs=-1, tree_method="hist"
)
light_model = lgb.LGBMRegressor(
    random_state=42, n_estimators=50, max_depth=5, n_jobs=-1, boosting_type="gbdt"
)
SPLITS = 3
kf = KFold(n_splits=SPLITS, shuffle=True, random_state=42)

# Dictionary to store model results
model_results = {}


X_train.shape


# Dictionary to store model results
model_results = {}

# Train and evaluate each model
for name, model in tqdm(
    [
        ("Decision Tree", dt_model),
        ("XGBoost", xgb_model),
        ("LightGBM", light_model),
        ("Random Forest", rf_model),
    ],
    desc="Training Models",
):
    print(f"\nTraining {name}...")

    start_time = time.time()
    with tqdm(desc=f"Training {name}", leave=False) as pbar:
        model.fit(X_train, y_train)
    end_time = time.time()
    duration = end_time - start_time
    print(f"{name} training completed in {duration:.2f} seconds")

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Perform cross-validation with progress bar
    print("Performing cross-validation...")
    cv_scores = []

    for train_idx, val_idx in tqdm(kf.split(X), desc=f"{name} CV", total=SPLITS):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train and evaluate
        start_time = time.time()
        model.fit(X_cv_train, y_cv_train)
        end_time = time.time()
        duration = end_time - start_time
        cv_score = r2_score(y_cv_val, model.predict(X_cv_val))
        cv_scores.append(cv_score)

    cv_scores = np.array(cv_scores)

    # Store results
    model_results[name] = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std(),
    }

    print(f"{name} evaluation completed\n")


# Print results
print("\nModel Performance Comparison:")
print("-" * 80)
for model_name, metrics in model_results.items():
    print(f"\n{model_name}:")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"R2 Score: {metrics['R2']:.4f}")
    print(
        f"Cross-validation R2: {metrics['CV_R2_mean']:.4f} (±{metrics['CV_R2_std']:.4f})"
    )

r"""°°°
While all models were successfully trained and evaluated, the overall predictive performance, as indicated by R² scores, is low across all models.  

XGBoost and LightGBM demonstrated slightly better performance compared to Decision Tree and Random Forest, but the practical utility of these models in their current state is limited due to the low R² values.  

Significant data quality concerns, previously identified in our initial data exploration, are likely contributing to these suboptimal results.

* **Low R² Scores:**  All models exhibit very low R² scores (ranging from 0.0260 to 0.0411).  This indicates that these models, in their current configuration and with the given features, explain only a small percentage of the variance in the premium amount.
* **RMSE and MAE Values:** The RMSE values are around 850, and MAE values are around 650.  Given the scale of premium amounts in our dataset, these error magnitudes are considered big.

### Low R² Scores & AI ethics
An R² of 0.02-0.04 shows around 96-98% of the premium amount variation unexplained.

Possible causes of decreasing R² are missing features, simple models or inherent randomness in data. 

However in this case, with the data preparation, the issue is likely with data quality.

Deploying a model with low R² is a serious ethical oversight. This is because of the following reasons:

#### Accuracy

Low R² typically results in low accuracy, which is true in our case, looking at RMSE and MAE magnitudes. If the model is deployed and used to make AI-Augmented Decisions, we face the following risks:
1. Undercharging premiums
2. Overcharging premiums

In the first case, this leads to financial instability for the insurance company in the long run, and cause loss in profits
In the second case, this leads to low-risk individuals unfairly charged higher premiums, which may discourage them and push them away from signing a policy with the insurance company, causing loss in profits.

#### Explainability

Usually Tree-based models are praised for their interpretability. However, intepreting a model that is low R² is ineffective. For example, a detective that is given the physical crime scene to work with, vs a detective that is far away, only given a walkie-talkie from a bystander. 

The detective at the crime scene has all the important clues, like fingerprints, witness statements, layout of room etc. Because they have rich and relevant information, they can build and share a very accurate picture of what happened and predict the premium.

However, the detective with a walkie-talkie only gets limited, incomplete, and noisy data, making it very hard to form an accurate picture of what happened and make predictions.

Even if the detective (the random forest) has similar experience (hyperparameters) and similar ability to share ideas (intepretability features), interpretability is more useful with the detective at the crime scene vs the detective with the walkie-talkie.

Summing it up, interpreting a low R² model is ineffective, because it doesn't even have a clear picture of the relationship between the features and target variable to share with us. And this is bad because Insurance companies are required to justify their pricing to customers and regulators. 

Deploying such a poor model will make it hard to provide clear, justifiable reasons for premium calculations.

°°°"""

# Feature importance analysis for Random Forest
feature_importance = pd.DataFrame(
    {"feature": feature_cols, "importance": rf_model.feature_importances_}
)
feature_importance = feature_importance.sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
print("-" * 80)
print(feature_importance.head(10))

# Analyze prediction errors
rf_predictions = rf_model.predict(X_test)
prediction_errors = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": rf_predictions,
        "Error": abs(y_test - rf_predictions),
    }
)

r"""°°°

Taking a look at the feature importance, some red flags are visible. Namely, the unrealstic feature ranking having a disconnect from reality.

Previous Claims, as mentioned, is usually a much stronger predictor.

High importance of "Policy Start Date_days_since_min" is also suspect. While it could have some minor influence like the annual increase in premiums and inflation guards, its high ranking suggest an artificial pattern in the data generation.

The very low importance of Occupation_Unemployed is also weird, because Occupation does indirectly correlate with risk.

There is much to point out, but what is clear is that while our Random Forest model and potentially the rest of the models (since they have comparable evaluation metrics) can identify some sort of parttern in our data, just that the findings are likely artifacts of synthetic data generation.

°°°"""

print("\nError Analysis:")
print("-" * 80)
print("Error Distribution Statistics:")
print(prediction_errors["Error"].describe())

# Calculate error percentiles
error_percentiles = np.percentile(prediction_errors["Error"], [25, 50, 75, 90, 95, 99])
print("\nError Percentiles:")
for p, v in zip([25, 50, 75, 90, 95, 99], error_percentiles):
    print(f"{p}th percentile: {v:.2f}")

r"""°°°
**Mean Error:** On average, the model's predictions are off by approximately \$654.10. This indicates the typical magnitude of error in our predictions.

**Standard Deviation of Error:** The error values have a standard deviation of \$546.96. This relatively high standard deviation suggests a considerable variability in the prediction errors, meaning the model's accuracy is not consistent across all predictions.

**Minimum Error:** The smallest error observed is very close to zero (\$0.01), indicating some predictions are highly accurate.

**Maximum Error:** The largest error reaches \$4,268.10, highlighting that in some cases, the model's predictions can be significantly off.


**Moderate Average Error:** The mean error of \$654.10 is substantial in the context of insurance premiums. While it's an average, it suggests that, on a typical prediction, the model is not very precise.

**High Error Variability:** The large standard deviation (546.96) and the wide range between minimum and maximum errors (from 0.01 to 4,268.10) indicate that the model's performance is inconsistent. It performs well for some instances but poorly for others.

**Skewed Error Distribution:** The fact that the mean error (654.10) is higher than the median error (522.76) suggests that the error distribution is right-skewed. This means there are more instances with smaller errors, but the larger errors are significantly larger, pulling the mean upwards.

### Error Percentile Breakdown

To further understand the error distribution, we examined specific percentiles:

**90th Percentile (\$1,320.78):** 90% of the predictions are within \$1,320.78 of the actual premium. This means that 10% of predictions have errors exceeding this amount.

**95th Percentile (\$1,722.10):** 95% of the predictions are within \$1,722.10 of the actual premium.  This indicates that 5% of predictions have substantial errors.

**99th Percentile (\$2,704.09):** 99% of the predictions are within \$2,704.09 of the actual premium.  Even at the 99th percentile, the error is still considerable, suggesting that in the worst 1% of cases, the model's predictions are significantly inaccurate.

°°°"""
r"""°°°
### Feature engineering: scaling and dimensoniality reduction for linear and SVM models
°°°"""

# 1. Apply PCA to handle multicollinearity and create new features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Exclude original skewed columns
feature_cols = [
    col
    for col in df.columns
    if col
    not in ["Premium Amount", "Annual Income", "Previous Claims", "Premium Amount"]
]
# Scale the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Analysis:")
print("-" * 80)
print(f"Number of original features: {X_scaled.shape[1]}")
print(f"Number of components after PCA: {X_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

r"""°°°
### Linear & SVM models
°°°"""

# Linear & SVM model goes here
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR

# Split PCA data
X_pca_train, X_pca_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# Initialize Linear Regression model
lr_model = LinearRegression()

# Initialize SVR model
svr_model = LinearSVR()  # Using LinearSVR for faster training

# Train and evaluate Linear Regression
print("\nTraining Linear Regression...")
start_time = time.time()
lr_model.fit(X_pca_train, y_train)
end_time = time.time()
duration = end_time - start_time
print(f"Linear Regression training completed in {duration:.2f} seconds")

y_pred_lr = lr_model.predict(X_pca_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Performing cross-validation for Linear Regression...")
cv_scores_lr = cross_val_score(lr_model, X_pca, y, cv=kf, scoring="r2")

# Store Linear Regression results
model_results["Linear Regression"] = {
    "RMSE": rmse_lr,
    "MAE": mae_lr,
    "R2": r2_lr,
    "CV_R2_mean": cv_scores_lr.mean(),
    "CV_R2_std": cv_scores_lr.std(),
}
print(f"Linear Regression evaluation completed\n")


# Train and evaluate SVR
print("\nTraining Support Vector Regression...")
start_time = time.time()
svr_model.fit(X_pca_train, y_train)
end_time = time.time()
duration = end_time - start_time
print(f"Support Vector Regression training completed in {duration:.2f} seconds")

y_pred_svr = svr_model.predict(X_pca_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("Performing cross-validation for Support Vector Regression...")
cv_scores_svr = cross_val_score(svr_model, X_pca, y, cv=kf, scoring="r2")


# Store SVR results
model_results["Support Vector Regression"] = {
    "RMSE": rmse_svr,
    "MAE": mae_svr,
    "R2": r2_svr,
    "CV_R2_mean": cv_scores_svr.mean(),
    "CV_R2_std": cv_scores_svr.std(),
}
print(f"Support Vector Regression evaluation completed\n")


r"""°°°
## Performance Measurement
°°°"""

# Print results
print("\nModel Performance Comparison:")
print("-" * 80)
for model_name, metrics in model_results.items():
    print(f"\n{model_name}:")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"R2 Score: {metrics['R2']:.4f}")
    print(
        f"Cross-validation R2: {metrics['CV_R2_mean']:.4f} (±{metrics['CV_R2_std']:.4f})"
    )

r"""°°°

Linear regression has the lowest R2 of 0.0028, which is significantly lower than even the worst-performing tree-based models. Linear regression also has the highest RMSE. This shows that the relationship between the feature and target variable is not linear, and we can benefit from more complex models.

However, we have interesting results with Linear SVR. A negative R2 means that the model is performing worse than guessing the mean of the target variable. Linear SVR also has the highest RMSE by a significant margin. Paradoxically, it has the lowest MAE among all the models. It suggests that SVR is making some very large errors, which drive up the RMSE, but on average, the absolute errors are smaller than other models. 

Regardless, the combination of PCA and linear models seems to not be a good fit.

#### Model Selection Verdict
None of these models perform well for real-world deployment. However, for the purpose of moving forward and choosing one model, the **XGBoost** is the least worst option, given the fact that is has higher R2 and lower RMSE and MAE.


°°°"""
r"""°°°
## Hyperparameter Tuning

We will utilize RandomizedCVSearch to find the best hyperparameters. It is similar to GridSearchCV but it is much quicker and arrives at very similar results.

Here are the hyperparameters we're adjusting, and how it impacts the model performance.
°°°"""

# Hyperparameter tuning for XGBoost
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define the parameter space
xgb_param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 7),
    "gamma": uniform(0, 0.5),
}

# Initialize XGBoost model for tuning
xgb_model = xgb.XGBRegressor(random_state=42, tree_method="hist")

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_dist,
    n_iter=50,  # Number of parameter settings sampled
    scoring="r2",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

print("\nPerforming RandomizedSearchCV for XGBoost...")
random_search.fit(X_train, y_train)

# Print best parameters and score
print("\nBest parameters found:")
print(random_search.best_params_)
print(f"\nBest cross-validation R2 score: {random_search.best_score_:.4f}")

# Use best model for predictions
best_xgb = random_search.best_estimator_

r"""°°°
## Conclusion

Perhaps one of the biggest problems in the notebook is the data quality. This has resulted in my inability to confidently recommend a model for production usage. However, I also recognize that the purpose of the assignment is not to build a production-ready model. Rather to explore the ethical dimensions of machine learning, and perhaps to explore how using synthetic data to model data for insurance predictions can be unethical. 

Regardless, in our notebook, we have seen additional discussion in how certain conditions in the machine learning pipeline can lead to AI ethical considerations, with some examples being missing and zero values, outliers, tree-based models vs linear models and low R2. 

In the context of insurance, synthetic data patterns that deviate from real-world insurance relations, leading to low model performance which could lead to unfair premium calcualtions, poor model interpretability making it difficult to justify premium decisions.

The best next step I can recommend for the insurance company, is to obtain real-world insurace data with proper documentation procedures, with robust data vaildation processes. Even if the model is able to learn the features of the model, the interpretability of the model is still important in insurance, where the customer or regulator can demand an explanation for the premium pricing anytime. Additionally, its important for the public documentation of how the predicted premium is used in AI-Augmented generated decision-making.

Given that the model is unable to do any of this, sometimes, the most ethical decision is not to deploy and use any models. It's better this way, than to risk implementing a system that could perpetuate unfair practices or cause harm to customers.



This analysis reveals critical insights about both the technical and ethical dimensions of applying machine learning to insurance premium prediction.

### Technical Findings

1. **Model Performance**
   - All models achieved low R² scores (0.02-0.04), explaining only 2-4% of premium variance
   - XGBoost performed marginally better but still inadequate for real-world use
   - Hyperparameter tuning yielded minimal improvements, suggesting fundamental data issues

2. **Data Quality Concerns**
   - Synthetic data patterns deviate from real-world insurance relationships
   - Unrealistic feature importance rankings (e.g., Previous Claims having low importance)
   - Suspicious uniformity in categorical distributions
   - Missing expected correlations between key variables

### Ethical Implications

1. **Fairness and Bias**
   - Low model performance could lead to unfair premium calculations
   - Risk of systematic bias against certain demographic groups
   - Potential for both overcharging and undercharging customers

2. **Transparency and Accountability**
   - Poor model interpretability makes it difficult to justify premium decisions
   - Challenges in meeting regulatory requirements for explainable pricing
   - Risk of eroding customer trust due to unexplainable premium variations

3. **Business Impact**
   - Financial instability risk from inaccurate premium calculations
   - Potential loss of customers due to unfair pricing
   - Regulatory compliance challenges
   - Reputational risks from deploying unreliable models

### Recommendations

1. **Data Quality**
   - Obtain real-world insurance data with proper documentation
   - Implement robust data validation processes
   - Ensure transparent labeling of synthetic data

2. **Model Development**
   - Establish minimum performance thresholds for model deployment
   - Incorporate domain expertise in feature engineering
   - Develop comprehensive model validation frameworks

3. **Ethical Guidelines**
   - Create clear policies for model deployment criteria
   - Establish regular bias auditing processes
   - Implement transparent communication protocols for premium calculations

### Final Verdict

While this exercise provided valuable insights into the machine learning pipeline for insurance premium prediction, the models developed are **NOT SUITABLE FOR DEPLOYMENT**. The combination of poor performance metrics and significant ethical concerns makes deployment irresponsible and potentially harmful to both the business and its customers.

This project serves as a crucial reminder that in the field of AI ethics, sometimes the most ethical decision is to **not deploy a model** when it fails to meet basic performance and fairness standards. It's better to acknowledge these limitations than to risk implementing a system that could perpetuate unfair practices or cause harm to customers.
°°°"""

