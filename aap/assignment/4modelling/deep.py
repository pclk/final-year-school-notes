import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import os
from tqdm import tqdm

# Constants
INPUT_FILE = "processed_data.npz"
MODEL_DIR = "saved_model"  # Directory to save the model
RANDOM_STATE = 42
BATCH_SIZE = 64  # Increased batch size for better gradient estimates
LEARNING_RATE = 0.0001  # lower learning rate by x10^-1 to avoid loss: nan
NUM_EPOCHS = 300  # More epochs for better convergence

# Model Architecture Constants
INITIAL_DENSE = 2048  # Base size for initial expansion
RESIDUAL_DENSE = 2048  # Must match INITIAL_DENSE for residual connections
PARALLEL_DENSE = 2048  # Same size for parallel paths
DENSE_BLOCK1 = 2048  # Must match combined parallel paths (PARALLEL_DENSE * 2)
DENSE_BLOCK2 = 2048  # Maintaining size through the network
REDUCTION_DENSE1 = 1024  # Start reducing dimensions
REDUCTION_DENSE2 = 512  # Further reduction
REDUCTION_DENSE3 = 256  # Final reduction before output
LEAKY_ALPHA = 0.2  # LeakyReLU alpha value

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Set random seeds for reproducibility
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def create_model(input_dim):
    # Create a much more complex model with multiple residual paths
    inputs = tf.keras.Input(shape=(input_dim,))

    # Initial expansion
    x = layers.Dense(INITIAL_DENSE, kernel_initializer="he_normal")(inputs)
    x = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x)

    # First residual block
    x1_1 = layers.Dense(RESIDUAL_DENSE, kernel_initializer="he_normal")(x)
    x1_1 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x1_1)
    x1_2 = layers.Dense(RESIDUAL_DENSE, kernel_initializer="he_normal")(x1_1)
    x1_2 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x1_2)
    x1 = layers.Add()([x, x1_2])

    # Parallel path 1
    p1 = layers.Dense(PARALLEL_DENSE, kernel_initializer="he_normal")(x1)
    p1 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(p1)

    # Second residual block
    x2_1 = layers.Dense(RESIDUAL_DENSE, kernel_initializer="he_normal")(x1)
    x2_1 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x2_1)
    x2_2 = layers.Dense(RESIDUAL_DENSE, kernel_initializer="he_normal")(x2_1)
    x2_2 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x2_2)
    x2 = layers.Add()([x1, x2_2])

    # Parallel path 2
    p2 = layers.Dense(PARALLEL_DENSE, kernel_initializer="he_normal")(x2)
    p2 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(p2)

    # Combine parallel paths
    combined = layers.Concatenate()([p1, p2])

    # Dense block 1
    d1 = layers.Dense(DENSE_BLOCK1, kernel_initializer="he_normal")(combined)
    d1 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(d1)

    # Third residual block
    x3_1 = layers.Dense(RESIDUAL_DENSE, kernel_initializer="he_normal")(d1)
    x3_1 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x3_1)
    x3_2 = layers.Dense(RESIDUAL_DENSE, kernel_initializer="he_normal")(x3_1)
    x3_2 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x3_2)
    x3 = layers.Add()([d1, x3_2])

    # Dense block 2
    d2 = layers.Dense(DENSE_BLOCK2, kernel_initializer="he_normal")(x3)
    d2 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(d2)

    # Fourth residual block
    x4_1 = layers.Dense(DENSE_BLOCK2, kernel_initializer="he_normal")(d2)
    x4_1 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x4_1)
    x4_2 = layers.Dense(DENSE_BLOCK2, kernel_initializer="he_normal")(x4_1)
    x4_2 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(x4_2)
    x4 = layers.Add()([d2, x4_2])

    # Dense reduction blocks
    d3 = layers.Dense(REDUCTION_DENSE1, kernel_initializer="he_normal")(x4)
    d3 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(d3)

    d4 = layers.Dense(REDUCTION_DENSE2, kernel_initializer="he_normal")(d3)
    d4 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(d4)

    d5 = layers.Dense(REDUCTION_DENSE3, kernel_initializer="he_normal")(d4)
    d5 = layers.LeakyReLU(alpha=LEAKY_ALPHA)(d5)

    # Final output
    outputs = layers.Dense(1)(d5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # Custom progress bar callback
    class ProgressCallback(callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.progress_bar = None

        def on_train_begin(self, logs=None):
            self.progress_bar = tqdm(total=NUM_EPOCHS, desc="Training")

        def on_epoch_end(self, epoch, logs=None):
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(
                {"loss": f"{logs['loss']:.4f}", "val_loss": f"{logs['val_loss']:.4f}"}
            )

        def on_train_end(self, logs=None):
            self.progress_bar.close()

    # Simple compilation with MSE loss
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse")

    # Train model
    print("\nTraining deep learning model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ProgressCallback()],
        verbose=0,
    )

    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test, verbose=0)

    return y_pred.flatten(), history


def main():
    # Load preprocessed data with binarized features
    print("Loading preprocessed data...")
    data = np.load(INPUT_FILE, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"]

    print(f"\nFeature information:")
    print(f"Total number of features: {X.shape[1]}")
    print(
        f"Number of soft skills: {len([f for f in feature_names if f.startswith('soft_')])}"
    )
    print(
        f"Number of hard skills: {len([f for f in feature_names if f.startswith('hard_')])}"
    )
    print(
        f"Number of fields: {len([f for f in feature_names if f.startswith('field_')])}"
    )
    data = np.load(INPUT_FILE, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    # Check for and handle any infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Print data statistics
    print("\nData Statistics:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X contains NaN: {np.isnan(X).any()}")
    print(f"y contains NaN: {np.isnan(y).any()}")
    print(f"X contains inf: {np.isinf(X).any()}")
    print(f"y contains inf: {np.isinf(y).any()}")
    print(f"X range: [{X.min()}, {X.max()}]")
    print(f"y range: [{y.min()}, {y.max()}]")

    # Additional data validation
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        raise ValueError("Data contains non-finite values after preprocessing!")

    # Split indices
    np.random.seed(RANDOM_STATE)
    indices = np.random.permutation(len(X))
    train_idx = indices[: int(0.7 * len(X))]
    val_idx = indices[int(0.7 * len(X)) : int(0.8 * len(X))]
    test_idx = indices[int(0.8 * len(X)) :]

    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Initialize model
    input_dim = X.shape[1]
    model = create_model(input_dim)

    # Train and evaluate model
    y_pred, history = train_and_evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Transform predictions back to original scale
    y_pred_orig = np.expm1(y_pred)
    y_test_orig = np.expm1(y_test)

    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

    print(f"\nDeep Learning Model Performance:")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # Print sample predictions
    print("\nSample Predictions:")
    n_samples = 5
    sample_indices = np.random.choice(len(y_test_orig), n_samples, replace=False)
    comparison = pd.DataFrame(
        {
            "Actual": y_test_orig[sample_indices],
            "Predicted": y_pred_orig[sample_indices],
        }
    )
    comparison["Difference"] = comparison["Predicted"] - comparison["Actual"]
    comparison["Percentage Error"] = abs(
        (comparison["Predicted"] - comparison["Actual"]) / comparison["Actual"] * 100
    )
    print(comparison.round(2))

    # Save the model in SavedModel (.pb) format for GCP deployment
    MODEL_DIR = "saved_model"
    VERSION = "1"
    EXPORT_PATH = os.path.join(MODEL_DIR, VERSION)

    print(f"\nSaving model in SavedModel format to: {EXPORT_PATH}")
    tf.saved_model.save(model, EXPORT_PATH)
    print("Model saved successfully!")

    print("\nTraining History:")
    for metric in history.history.keys():
        print(f"Final {metric}: {history.history[metric][-1]:.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
