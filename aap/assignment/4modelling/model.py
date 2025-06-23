# Constants
INPUT_FILE = "../3eda/cleaned_jobs.csv"
OUTPUT_FILE = "forest.pkl"
PROCESSED_DATA_FILE = "processed_data.npz"
RANDOM_STATE = 42

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tqdm import tqdm
import os

# Load the original data regardless of preprocessing
print("Loading original data...")
df = pd.read_csv(INPUT_FILE)

# Check if processed data exists
if os.path.exists(PROCESSED_DATA_FILE):
    print(f"Loading preprocessed data from {PROCESSED_DATA_FILE}...")
    data = np.load(PROCESSED_DATA_FILE, allow_pickle=True)
    X = pd.DataFrame(data["X"], columns=data["feature_names"])
    y = data["y"]
    label_encoders = data["label_encoders"].item()
    soft_kmeans = data["soft_kmeans"].item()
    hard_kmeans = data["hard_kmeans"].item()
    field_kmeans = data["field_kmeans"].item()
    soft_vectorizer = data["soft_vectorizer"].item()
    hard_vectorizer = data["hard_vectorizer"].item()
    field_vectorizer = data["field_vectorizer"].item()
    soft_skill_clusters = data["soft_skill_clusters"].item()
    hard_skill_clusters = data["hard_skill_clusters"].item()
    field_clusters = data["field_clusters"].item()
    print("Preprocessed data loaded successfully!")

# Create feature matrix X
print("\nPreparing features...")


# Function to convert list strings to actual lists
def parse_list_string(s):
    if pd.isna(s):
        return []
    try:
        # Remove brackets and split by comma
        return [x.strip().strip("'\"") for x in s.strip("[]").split(",") if x.strip()]
    except:
        return []


# Parse skill columns
df["soft_skills"] = df["soft_skills"].apply(parse_list_string)
df["hard_skills"] = df["hard_skills"].apply(parse_list_string)
df["field_of_study"] = df["field_of_study"].apply(parse_list_string)


# Function to cluster skills
def cluster_skills(skills_list, n_clusters=100, prefix="cluster"):
    if not skills_list:
        return {}, None, None

    # Convert skills to strings for TF-IDF
    skills_text = [" ".join(skill.lower().split("_")) for skill in skills_list]

    # Create TF-IDF vectors
    print(f"\nVectorizing {prefix} skills...")
    vectorizer = TfidfVectorizer(stop_words="english")
    skill_vectors = vectorizer.fit_transform(skills_text)

    # Perform clustering
    print(f"Clustering {prefix} skills...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(skill_vectors)

    # Create skill to cluster mapping
    skill_clusters = {skill: cluster for skill, cluster in zip(skills_list, clusters)}

    return skill_clusters, kmeans, vectorizer


# Get all unique skills
all_soft_skills = set()
all_hard_skills = set()
all_fields = set()

for skills in df["soft_skills"]:
    all_soft_skills.update(skills)
for skills in df["hard_skills"]:
    all_hard_skills.update(skills)
for fields in df["field_of_study"]:
    all_fields.update(fields)

print(f"Total unique soft skills: {len(all_soft_skills)}")
print(f"Total unique hard skills: {len(all_hard_skills)}")
print(f"Total unique fields: {len(all_fields)}")

# Cluster the skills
N_CLUSTERS = 100
soft_skill_clusters, soft_kmeans, soft_vectorizer = cluster_skills(
    list(all_soft_skills), N_CLUSTERS, "soft"
)
hard_skill_clusters, hard_kmeans, hard_vectorizer = cluster_skills(
    list(all_hard_skills), N_CLUSTERS, "hard"
)
field_clusters, field_kmeans, field_vectorizer = cluster_skills(
    list(all_fields), N_CLUSTERS, "field"
)


# Create clustered skill features
def create_skill_features(row, skill_clusters, n_clusters, prefix):
    if not skill_clusters:
        return {}

    # Initialize cluster counts
    cluster_counts = np.zeros(n_clusters)

    # Count skills in each cluster
    for skill in row:
        if skill in skill_clusters:
            cluster_counts[skill_clusters[skill]] += 1

    # Create features
    return {f"{prefix}_cluster_{i}": count for i, count in enumerate(cluster_counts)}


# Create all feature dictionaries
print("Creating skill features...")
feature_dicts = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    features = {}

    # Add clustered skill features
    features.update(
        create_skill_features(
            row["soft_skills"], soft_skill_clusters, N_CLUSTERS, "soft"
        )
    )
    features.update(
        create_skill_features(
            row["hard_skills"], hard_skill_clusters, N_CLUSTERS, "hard"
        )
    )
    features.update(
        create_skill_features(
            row["field_of_study"], field_clusters, N_CLUSTERS, "field"
        )
    )

    # Add other numerical/categorical features and derived features
    features["min_years_experience"] = row["min_years_experience"]
    features["total_skills"] = len(row["soft_skills"]) + len(
        row["hard_skills"]
    )  # Total skill count
    features["skills_per_year"] = features["total_skills"] / (
        row["min_years_experience"] + 1
    )  # Skill density

    # Add categorical features with label encoding
    for col in [
        "country",
        "location_flexibility",
        "contract_type",
        "education_level",
        "seniority",
    ]:
        if pd.notna(row[col]):
            features[col] = row[col]
        else:
            features[col] = "unknown"

    feature_dicts.append(features)

# Convert to DataFrame
X = pd.DataFrame(feature_dicts)

# Label encode categorical columns
categorical_cols = [
    "country",
    "location_flexibility",
    "contract_type",
    "education_level",
    "seniority",
]
label_encoders = {}

for col in tqdm(categorical_cols, desc="Encoding categorical columns"):
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Target variable y (yearly salary midpoint) with log transform
y = np.log1p(df["yearly_salary_midpoint"])  # log1p handles zero values

# Add some basic data validation
print("\nValidating salary data...")
print(f"Salary statistics before transformation:")
print(f"Min: ${df['yearly_salary_midpoint'].min():,.2f}")
print(f"Max: ${df['yearly_salary_midpoint'].max():,.2f}")
print(f"Mean: ${df['yearly_salary_midpoint'].mean():,.2f}")
print(f"Median: ${df['yearly_salary_midpoint'].median():,.2f}")

# Remove extreme outliers (outside 3 standard deviations)
z_scores = np.abs((y - y.mean()) / y.std())
y_clean = y[z_scores < 3]
X_clean = X[z_scores < 3]

print(f"\nRemoved {len(y) - len(y_clean)} outliers")
y = y_clean
X = X_clean

# Remove any rows where target is NaN
valid_mask = ~y.isna()
X = X[valid_mask]
y = y[valid_mask]

print(f"\nFinal dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Save processed data if we just created it
if not os.path.exists(PROCESSED_DATA_FILE):
    print(f"\nSaving processed data to {PROCESSED_DATA_FILE}...")
    np.savez(
        PROCESSED_DATA_FILE,
        X=X.values,
        y=y.values,
        feature_names=X.columns,
        label_encoders=label_encoders,
        soft_kmeans=soft_kmeans,
        hard_kmeans=hard_kmeans,
        field_kmeans=field_kmeans,
        soft_vectorizer=soft_vectorizer,
        hard_vectorizer=hard_vectorizer,
        field_vectorizer=field_vectorizer,
        soft_skill_clusters=soft_skill_clusters,
        hard_skill_clusters=hard_skill_clusters,
        field_clusters=field_clusters,
    )
    print("Processed data saved successfully!")

# Split the data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add interaction terms for numerical features
print("\nAdding interaction terms...")
numerical_features = [
    "min_years_experience",
    "total_skills",
    "skills_per_year",
]

# Create interaction features for both train and test sets
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Add interactions between numerical features
for i in range(len(numerical_features)):
    for j in range(i + 1, len(numerical_features)):
        feat1, feat2 = numerical_features[i], numerical_features[j]
        interaction_name = f"{feat1}_{feat2}_interaction"
        X_train_df[interaction_name] = X_train_df[feat1] * X_train_df[feat2]
        X_test_df[interaction_name] = X_test_df[feat1] * X_test_df[feat2]

# Scale all features
print("\nScaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

# Keep track of feature names
feature_names = X_train_df.columns

# Train Random Forest with better parameters
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Fit and get feature importances
rf_model.fit(X_train, y_train)

# Print feature importances before proceeding
print("\nInitial Feature Importances:")
feature_imp = pd.DataFrame(
    {"feature": feature_names, "importance": rf_model.feature_importances_}
)
feature_imp = feature_imp.sort_values("importance", ascending=False)
print(feature_imp.head(10).to_string(index=False))

# Check for potential data issues
print("\nData Diagnostics:")
print(f"Training set shape: {X_train.shape}")
print(f"Target value range: {y_train.min():.2f} to {y_train.max():.2f}")
print("\nFeature statistics:")
X_df = pd.DataFrame(X_train, columns=feature_names)
print(X_df.describe().round(2))

# Check for any infinite or NaN values
print("\nInfinite values in features:", np.any(np.isinf(X_train)))
print("NaN values in features:", np.any(np.isnan(X_train)))
print("NaN values in target:", np.any(np.isnan(y_train)))

# Make predictions
print("\nEvaluating model...")
y_pred = rf_model.predict(X_test)

# Transform predictions back to original scale
y_pred_orig = np.expm1(y_pred)
y_test_orig = np.expm1(y_test)

# Calculate metrics
mse = mean_squared_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, y_pred_orig)
mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

print(f"\nModel Performance:")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Print some actual vs predicted examples
print("\nSample Predictions:")
n_samples = 5
# Convert to numpy arrays if they aren't already
y_test_orig_array = np.array(y_test_orig)
y_pred_orig_array = np.array(y_pred_orig)

sample_indices = np.random.choice(len(y_test_orig_array), n_samples, replace=False)
comparison = pd.DataFrame(
    {
        "Actual": y_test_orig_array[sample_indices],
        "Predicted": y_pred_orig_array[sample_indices],
    }
)
comparison["Difference"] = comparison["Predicted"] - comparison["Actual"]
comparison["Percentage Error"] = abs(
    (comparison["Predicted"] - comparison["Actual"]) / comparison["Actual"] * 100
)
print(comparison.round(2))

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame(
    {"feature": feature_names, "importance": rf_model.feature_importances_}
)
feature_importance = feature_importance.sort_values("importance", ascending=False)
print(feature_importance.head(10).to_string(index=False))

# Save the model and label encoders
print(f"\nSaving model to {OUTPUT_FILE}...")
joblib.dump(
    {
        "model": rf_model,
        "label_encoders": label_encoders,
        "feature_columns": X.columns.tolist(),
        "soft_kmeans": soft_kmeans,
        "hard_kmeans": hard_kmeans,
        "field_kmeans": field_kmeans,
        "soft_vectorizer": soft_vectorizer,
        "hard_vectorizer": hard_vectorizer,
        "field_vectorizer": field_vectorizer,
        "soft_skill_clusters": soft_skill_clusters,
        "hard_skill_clusters": hard_skill_clusters,
        "field_clusters": field_clusters,
    },
    OUTPUT_FILE,
)

print("Done!")
