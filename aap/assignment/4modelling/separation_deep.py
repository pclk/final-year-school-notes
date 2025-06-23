import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import os

# Constants
INPUT_FILE = "../3eda/cleaned_jobs.csv"
WORD2VEC_SIZE = 100
RANDOM_STATE = 42
COUNTRIES = ["US", "IN", "SG"]

# Create output directories
os.makedirs("country_models", exist_ok=True)


def parse_list_string(s):
    if pd.isna(s):
        return []
    try:
        return [x.strip().strip("'\"") for x in s.strip("[]").split(",") if x.strip()]
    except:
        return []


def preprocess_text(text):
    if pd.isna(text):
        return []
    words = str(text).lower().split()
    words = [w.strip(".,!?()[]{}:;") for w in words if len(w) > 2]
    return words


def create_sequence_vector(sequence, w2v_model):
    vectors = [w2v_model.wv[token] for token in sequence if token in w2v_model.wv]
    if not vectors:
        return np.zeros(WORD2VEC_SIZE)
    return np.mean(vectors, axis=0)


def train_country_model(country_df, country_code):
    print(f"\n=== Training model for {country_code} ===")
    print(f"Dataset size: {len(country_df)} records")

    # Handle salary distribution
    if country_code == "IN":
        # Remove extreme outliers using quantiles
        Q1 = country_df["yearly_salary_midpoint"].quantile(0.05)
        Q3 = country_df["yearly_salary_midpoint"].quantile(0.95)

        print(f"\nBefore cleaning - Records: {len(country_df)}")
        print(
            f"Salary range: {country_df['yearly_salary_midpoint'].min():,.2f} to {country_df['yearly_salary_midpoint'].max():,.2f}"
        )

        # Filter out extreme values
        country_df = country_df[
            (country_df["yearly_salary_midpoint"] >= Q1)
            & (country_df["yearly_salary_midpoint"] <= Q3)
        ]

        print(f"After cleaning - Records: {len(country_df)}")
        print(
            f"Salary range: {country_df['yearly_salary_midpoint'].min():,.2f} to {country_df['yearly_salary_midpoint'].max():,.2f}"
        )

    # Prepare sequences for Word2Vec
    print("Preparing sequences...")
    all_sequences = []
    for idx, row in tqdm(country_df.iterrows(), total=len(country_df)):
        sequence = []

        # Add skills and fields
        sequence.extend([f"soft_{skill.lower()}" for skill in row["soft_skills"]])
        sequence.extend([f"hard_{skill.lower()}" for skill in row["hard_skills"]])
        sequence.extend([f"field_{field.lower()}" for field in row["field_of_study"]])

        # Process text fields
        sequence.extend(
            [f"desc_{word}" for word in preprocess_text(row["job_description"])]
        )
        sequence.extend([f"title_{word}" for word in preprocess_text(row["job_title"])])
        sequence.extend([f"query_{word}" for word in preprocess_text(row["query"])])

        if pd.notna(row["location"]):
            sequence.append(f"loc_{str(row['location']).lower()}")

        # Add categorical features
        for col in [
            "location_flexibility",
            "contract_type",
            "education_level",
            "seniority",
        ]:
            if pd.notna(row[col]):
                sequence.append(f"{col}_{str(row[col]).lower()}")

        all_sequences.append(sequence)

    # Try to load existing Word2Vec model, train new one if it doesn't exist
    model_path = f"country_models/{country_code}/word2vec_model.bin"
    if os.path.exists(model_path):
        print(f"Loading existing Word2Vec model from {model_path}...")
        w2v_model = Word2Vec.load(model_path)
    else:
        print("Training new Word2Vec model...")
        w2v_model = Word2Vec(
            sentences=all_sequences,
            vector_size=WORD2VEC_SIZE,
            window=5,
            min_count=2,
            workers=8,
            sg=1,
            negative=5,
            epochs=5,
        )

    # Create feature vectors
    print("Creating feature vectors...")
    X = np.array(
        [create_sequence_vector(seq, w2v_model) for seq in tqdm(all_sequences)]
    )

    # Add numerical features
    numerical_features = ["min_years_experience"]
    scaler = StandardScaler()
    for feature in numerical_features:
        feature_values = country_df[feature].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(feature_values)
        X = np.hstack((X, scaled_values))

    # Prepare target with robust scaling for IN
    if country_code == "IN":
        # Double log transform for extreme skewness
        y = np.log1p(np.log1p(country_df["yearly_salary_midpoint"].values))
    else:
        # Regular log transform for other countries
        y = np.log1p(country_df["yearly_salary_midpoint"].values)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Build model
    print("Building and training model...")
    model = Sequential(
        [
            Dense(1024, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(512, activation="relu"),
            Dropout(0.2),
            Dense(256, activation="relu"),
            Dropout(0.1),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=1,
    )

    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test, verbose=0)

    # Transform predictions back
    if country_code == "IN":
        # Reverse double log transform for IN
        y_pred_orig = np.expm1(np.expm1(y_pred))
        y_test_orig = np.expm1(np.expm1(y_test))
    else:
        # Regular inverse transform for other countries
        y_pred_orig = np.expm1(y_pred)
        y_test_orig = np.expm1(y_test)

    # Get test indices
    test_indices = country_df.index[len(X_train) :]

    # Create predictions DataFrame
    test_df = country_df.loc[test_indices].copy()
    test_df["predicted_salary"] = y_pred_orig
    test_df["actual_salary"] = y_test_orig
    test_df["prediction_error"] = test_df["predicted_salary"] - test_df["actual_salary"]
    test_df["prediction_error_percent"] = (
        test_df["prediction_error"] / test_df["actual_salary"]
    ) * 100

    # Take random subset of 100 predictions (or less if test set is smaller)
    sample_size = min(100, len(test_df))
    sample_df = test_df.sample(n=sample_size, random_state=RANDOM_STATE)

    # Save predictions to CSV
    sample_df.to_csv(f"{model_dir}/test_predictions_sample.csv", index=False)
    print(
        f"\nSaved {sample_size} sample predictions to {model_dir}/test_predictions_sample.csv"
    )

    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

    print(f"\nModel Performance for {country_code}:")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # Save models
    model_dir = f"country_models/{country_code}"
    os.makedirs(model_dir, exist_ok=True)

    w2v_model.save(f"{model_dir}/word2vec_model.bin")
    model.save(f"{model_dir}/deep_model.h5")
    joblib.dump(scaler, f"{model_dir}/scaler.joblib")

    return {"rmse": rmse, "r2": r2, "mape": mape, "n_samples": len(country_df)}


# Main execution
print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# Parse list columns
df["soft_skills"] = df["soft_skills"].apply(parse_list_string)
df["hard_skills"] = df["hard_skills"].apply(parse_list_string)
df["field_of_study"] = df["field_of_study"].apply(parse_list_string)

# Train separate models for each country
results = {}
for country in COUNTRIES:
    country_df = df[df["country"] == country].copy()
    if len(country_df) > 0:
        results[country] = train_country_model(country_df, country)

# Print comparative results
print("\n=== Comparative Results ===")
results_df = pd.DataFrame(results).T
results_df.columns = ["RMSE", "R²", "MAPE", "Sample Size"]
print(results_df)

# Save results
results_df.to_csv("country_models/comparative_results.csv")
print("\nDone! Models and results saved in 'country_models' directory")
