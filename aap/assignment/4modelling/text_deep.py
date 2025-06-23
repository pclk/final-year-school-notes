import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

# Constants
INPUT_FILE = "../3eda/cleaned_jobs.csv"
WORD2VEC_SIZE = 100
RANDOM_STATE = 42
MODEL_OUTPUT = "deep_salary_model.h5"
WORD2VEC_OUTPUT = "word2vec_model.bin"

# Load and prepare data
print("Loading data...")
df = pd.read_csv(INPUT_FILE)


def parse_list_string(s):
    if pd.isna(s):
        return []
    try:
        return [x.strip().strip("'\"") for x in s.strip("[]").split(",") if x.strip()]
    except:
        return []


# Parse skill columns
df["soft_skills"] = df["soft_skills"].apply(parse_list_string)
df["hard_skills"] = df["hard_skills"].apply(parse_list_string)
df["field_of_study"] = df["field_of_study"].apply(parse_list_string)


# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return []
    # Convert to lowercase and split
    words = str(text).lower().split()
    # Remove very short words and basic punctuation
    words = [w.strip(".,!?()[]{}:;") for w in words if len(w) > 2]
    return words


# Prepare text sequences for Word2Vec
print("Preparing sequences for Word2Vec...")
all_sequences = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    sequence = []

    # Add skills and fields as tokens
    sequence.extend([f"soft_{skill.lower()}" for skill in row["soft_skills"]])
    sequence.extend([f"hard_{skill.lower()}" for skill in row["hard_skills"]])
    sequence.extend([f"field_{field.lower()}" for field in row["field_of_study"]])

    # Process job description
    desc_words = preprocess_text(row["job_description"])
    sequence.extend([f"desc_{word}" for word in desc_words])

    # Process job title
    title_words = preprocess_text(row["job_title"])
    sequence.extend([f"title_{word}" for word in title_words])

    # Process query
    query_words = preprocess_text(row["query"])
    sequence.extend([f"query_{word}" for word in query_words])

    # Process location
    if pd.notna(row["location"]):
        sequence.append(f"loc_{str(row['location']).lower()}")

    # Add other categorical features
    for col in [
        "country",
        "location_flexibility",
        "contract_type",
        "education_level",
        "seniority",
    ]:
        if pd.notna(row[col]):
            sequence.append(f"{col}_{str(row[col]).lower()}")

    all_sequences.append(sequence)

# Train Word2Vec model with optimized parameters
print("Training Word2Vec model...")
w2v_model = Word2Vec(
    sentences=all_sequences,
    vector_size=WORD2VEC_SIZE,
    window=5,  # Reduced window size
    min_count=5,  # Increased minimum frequency to reduce vocabulary
    workers=8,  # Increased number of workers for faster processing
    sg=1,  # Skip-gram model
    negative=5,  # Reduced negative sampling
    epochs=5,  # Reduced number of epochs
    compute_loss=True,
)

# Save Word2Vec model
w2v_model.save(WORD2VEC_OUTPUT)

# Create feature vectors
print("Creating feature vectors...")


def create_sequence_vector(sequence):
    vectors = [w2v_model.wv[token] for token in sequence if token in w2v_model.wv]
    if not vectors:
        return np.zeros(WORD2VEC_SIZE)
    return np.mean(vectors, axis=0)


X = np.array([create_sequence_vector(seq) for seq in tqdm(all_sequences)])

# Add numerical features
numerical_features = ["min_years_experience"]
for feature in numerical_features:
    feature_values = df[feature].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_values)
    X = np.hstack((X, scaled_values))

# Prepare target variable
y = np.log1p(df["yearly_salary_midpoint"].values)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Build deep learning model
print("Building deep learning model...")
model = Sequential(
    [
        Dense(2048, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(2048, activation="relu"),
        Dropout(0.2),
        Dense(1048, activation="relu"),
        Dropout(0.1),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1),
    ]
)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# Train model
print("Training deep learning model...")
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
)

# Evaluate model
print("\nEvaluating model...")
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}")

# Make predictions and evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test, verbose=0)

# Transform predictions back to original scale
y_pred_orig = np.expm1(y_pred)
y_test_orig = np.expm1(y_test)

# Calculate comprehensive metrics
mse = mean_squared_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_orig, y_pred_orig)
mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

print(f"\nDeep Learning Model Performance:")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Print training history
print("\nTraining History:")
for metric in history.history.keys():
    print(f"Final {metric}: {history.history[metric][-1]:.4f}")

# Print sample predictions
print("\nSample Predictions:")
n_samples = 5
sample_indices = np.random.choice(len(y_test_orig), n_samples, replace=False)
comparison = pd.DataFrame(
    {
        "Actual": y_test_orig[sample_indices],
        "Predicted": y_pred_orig[sample_indices].flatten(),
    }
)
comparison["Difference"] = comparison["Predicted"] - comparison["Actual"]
comparison["Percentage Error"] = abs(
    (comparison["Predicted"] - comparison["Actual"]) / comparison["Actual"] * 100
)
print(comparison.round(2))

# Save the model in .pb format
MODEL_PB_DIR = "salary_model_pb"
print(f"\nSaving model to {MODEL_PB_DIR} in .pb format...")

# Convert the model to a concrete function
full_model = tf.function(lambda x: model(x))
concrete_func = full_model.get_concrete_function(
    tf.TensorSpec([None, X_train.shape[1]], tf.float32)
)

# Save the model in SavedModel format with .pb
tf.saved_model.save(
    model,
    MODEL_PB_DIR,
    signatures={
        "serving_default": tf.function(
            lambda x: {"output": model(x)}
        ).get_concrete_function(
            tf.TensorSpec([None, X_train.shape[1]], tf.float32, name="input")
        )
    },
)

# Save the Word2Vec parameters separately
print("Saving model parameters...")
model_params = {
    "input_shape": X_train.shape[1],
    "word2vec_size": WORD2VEC_SIZE,
    "numerical_features": numerical_features,
    "feature_scaler": scaler,
}
joblib.dump(model_params, "model_params.joblib")

print("Done! Model saved in .pb format")

# Verify the saved model
print("\nVerifying saved model...")
loaded_model = tf.saved_model.load(MODEL_PB_DIR)
print("Model loaded successfully!")

# Test inference with the loaded model
test_data = X_test[:1]
original_prediction = model.predict(test_data)

# Get prediction from loaded model
serving_fn = loaded_model.signatures["serving_default"]
loaded_prediction = serving_fn(tf.constant(test_data, dtype=tf.float32))
prediction_value = loaded_prediction["output"].numpy()

print("\nPrediction test:")
print(f"Original model prediction: {original_prediction[0][0]:.4f}")
print(f"Loaded model prediction:   {prediction_value[0][0]:.4f}")
