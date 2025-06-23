r"""°°°
### Sub task 2 - Model Development with PyCaret
°°°"""
# |%%--%%| <OS1zzWc26D|FAUd5NiNqz>

from pycaret.regression import *

# Prepare data for PyCaret
# Combine features and target into a single dataframe for training
train_data = train_df[feature_cols + [target]].copy()
val_data = val_df[feature_cols + [target]].copy()

# Log dataset information
log("pycaret_train_samples", len(train_data))
log("pycaret_val_samples", len(val_data))
log("pycaret_features", len(feature_cols))
log("pycaret_train_cols", list(train_data))

# |%%--%%| <FAUd5NiNqz|XSs7Ij4gvg>

# Initialize PyCaret setup with MLflow tracking
# Note: PyCaret automatically logs to MLflow when log_experiment=True
reg_setup = setup(
    data=train_data,
    target=target,
    session_id=42,
    log_experiment=True,
    experiment_name="Housing Price Prediction",
    log_plots=True,
    verbose=True,
    ignore_features=["time_period", "price_bin", "strat_var"]
    if any(
        col in train_data.columns for col in ["time_period", "price_bin", "strat_var"]
    )
    else None,
    fold_strategy="timeseries",  # Use time series cross-validation
    data_split_shuffle=False,
    fold_shuffle=False,
    fold=5,
)

# Log setup parameters
log("pycaret_normalize", True)
log("pycaret_transformation", True)
log("pycaret_fold_strategy", "timeseries")
log("pycaret_folds", 5)

# |%%--%%| <XSs7Ij4gvg|Hncq0uIBMT>

# Compare models and get the best models table
best_models = compare_models(
    n_select=5,  # Select top 5 models
    sort="RMSE",  # Sort by RMSE
    verbose=True,
)

# If best_models is a single model (not a list)
if not isinstance(best_models, list):
    best_models = [best_models]

# |%%--%%| <Hncq0uIBMT|tPIWMF85b7>

# Tune each of the top models
tuned_models = []
for i, model in enumerate(best_models):
    model_name = model.__class__.__name__
    print(f"\nTuning {model_name}...")

    # Log the base model name
    log(f"top_model_{i+1}", model_name)

    # Tune the model
    tuned_model = tune_model(
        model,
        optimize="RMSE",
        n_iter=10,
        search_library="optuna",
    )

    tuned_models.append(tuned_model)

    # Evaluate on validation set
    pred_holdout = predict_model(tuned_model, data=val_data)
    val_rmse = np.sqrt(
        mean_squared_error(pred_holdout[target], pred_holdout["prediction_label"])
    )
    val_r2 = r2_score(pred_holdout[target], pred_holdout["prediction_label"])

    # Log validation metrics
    mlflow.log_metric(f"val_rmse_{model_name}", val_rmse)
    mlflow.log_metric(f"val_r2_{model_name}", val_r2)

    # Create and log residual plot
    plt.figure(figsize=(10, 6))
    residuals = pred_holdout[target] - pred_holdout["prediction_label"]
    plt.scatter(pred_holdout[target], residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - {model_name}")
    plt.tight_layout()
    plt.savefig(f"residuals_{model_name}.png")
    mlflow.log_artifact(f"residuals_{model_name}.png")

    # Create and log actual vs predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_holdout[target], pred_holdout["prediction_label"], alpha=0.5)
    plt.plot(
        [pred_holdout[target].min(), pred_holdout[target].max()],
        [pred_holdout[target].min(), pred_holdout[target].max()],
        "r--",
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.tight_layout()
    plt.savefig(f"actual_vs_predicted_{model_name}.png")
    mlflow.log_artifact(f"actual_vs_predicted_{model_name}.png")

# |%%--%%| <tPIWMF85b7|Ens6XZ4Kev>

# Select the best model based on validation RMSE
best_idx = np.argmin(
    [
        np.sqrt(
            mean_squared_error(
                predict_model(model, data=val_data)[target],
                predict_model(model, data=val_data)["prediction_label"],
            )
        )
        for model in tuned_models
    ]
)

best_model = tuned_models[best_idx]
best_model_name = best_model.__class__.__name__

# |%%--%%| <Ens6XZ4Kev|609xJgB2L9>

# Log the final best model
log("best_model", best_model_name)
final_model = finalize_model(best_model)


def predict(input_data):
    """
    Predict the original property price based on input data.

    Args:
        input_data: Dictionary or tuple of dictionaries containing property features

    Returns:
        Predicted property price(s) in original scale ($)
    """
    # Convert input data to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, tuple):
        input_df = pd.DataFrame(list(input_data))
    else:
        input_df = pd.DataFrame(input_data)

    # Apply necessary preprocessing to match the model's expected features
    processed_df = preprocess_for_prediction(input_df)

    # Generate predictions using the PyCaret model
    prediction_df = predict_model(final_model, data=processed_df)

    # Extract the transformed predictions
    predicted_transformed_values = prediction_df["prediction_label"].values

    # Inverse-transform to get original price
    original_prices = [
        inverse_transform_price(val) for val in predicted_transformed_values
    ]

    # Return single value or list based on input
    if len(original_prices) == 1:
        return original_prices[0]
    else:
        return original_prices


def preprocess_for_prediction(df):
    """
    Apply all necessary preprocessing steps to prepare data for prediction.

    Args:
        df: DataFrame with raw input features

    Returns:
        DataFrame with all features required by the model
    """
    processed = df.copy()

    # 1. Property Type encoding
    if "Type" in processed.columns:
        # Map text property types to codes if needed
        type_mapping = {"House": "h", "Unit/Apartment": "u", "Townhouse": "t"}
        if not processed["Type"].isin(["h", "u", "t"]).all():
            processed["Type"] = processed["Type"].map(lambda x: type_mapping.get(x, x))

        # Create one-hot encoded features
        processed["PropType_House"] = (processed["Type"] == "h").astype(int)
        processed["PropType_Townhouse"] = (processed["Type"] == "t").astype(int)
        processed["PropType_Unit/Apartment"] = (processed["Type"] == "u").astype(int)
        processed = processed.drop("Type", axis=1)

    # 2. Method encoding
    if "Method" in processed.columns:
        # Create one-hot encoded Method features
        method_columns = [
            "Method_PI",
            "Method_S",
            "Method_SA",
            "Method_SP",
            "Method_VB",
        ]
        for col in method_columns:
            method_code = col.split("_")[1]
            processed[col] = (processed["Method"] == method_code).astype(int)
        processed = processed.drop("Method", axis=1)

    # 3. Suburb encoding
    if "Suburb" in processed.columns:
        # Use the suburb_to_rank_dict from the transformer
        suburb_ranks = transform_suburb.encoder.suburb_to_rank_dict
        processed["Suburb_PriceRank"] = processed["Suburb"].map(
            suburb_ranks, na_action="ignore"
        )
        # Fill missing values with median rank
        if processed["Suburb_PriceRank"].isna().any():
            median_rank = np.median(list(suburb_ranks.values()))
            processed["Suburb_PriceRank"] = processed["Suburb_PriceRank"].fillna(
                median_rank
            )
        processed = processed.drop("Suburb", axis=1)

    # 4. Seller encoding
    if "Seller" in processed.columns:
        # Create one-hot encoded Seller features
        seller_cols = [c for c in df.columns if c.startswith("Seller_")]
        common_sellers = [
            "Barry",
            "Biggin",
            "Brad",
            "Buxton",
            "Fletchers",
            "Gary",
            "Greg",
            "Harcourts",
            "Hodges",
            "Jas",
            "Jellis",
            "Kay",
            "Love",
            "Marshall",
            "McGrath",
            "Miles",
            "Nelson",
            "Noel",
            "RT",
            "Raine",
            "Ray",
            "Stockdale",
            "Sweeney",
            "Village",
            "Williams",
            "Woodards",
            "YPA",
            "hockingstuart",
        ]

        # Initialize all seller columns to 0
        for seller in common_sellers:
            processed[f"Seller_{seller}"] = 0

        # Set the appropriate column to 1 or "Other" if not in common sellers
        for idx, seller in enumerate(processed["Seller"]):
            if seller in common_sellers:
                processed.loc[idx, f"Seller_{seller}"] = 1
            else:
                processed.loc[idx, "Seller_Other"] = 1

        processed = processed.drop("Seller", axis=1)

    # 5. Direction features if needed
    if "Direction" in processed.columns:
        direction_cols = ["Direction_N", "Direction_S", "Direction_E", "Direction_W"]
        for dir_col in direction_cols:
            dir_code = dir_col.split("_")[1]
            processed[dir_col] = (processed["Direction"] == dir_code).astype(int)
        processed = processed.drop("Direction", axis=1)

    # 6. Numerical transformations
    # Apply Box-Cox transformations to numeric features
    if "Landsize" in processed.columns:
        processed["Landsize_Transformed"] = box_cox_transform(
            processed["Landsize"],
            boxcox_store["landsize_lambda"],
            boxcox_store.get("landsize_offset", 0),
        )

    if "BuildingArea" in processed.columns:
        processed["BuildingArea_Transformed"] = box_cox_transform(
            processed["BuildingArea"],
            boxcox_store["building_area_lambda"],
            boxcox_store.get("building_area_offset", 0),
        )

    if "Distance" in processed.columns:
        processed["Distance_Transformed"] = box_cox_transform(
            processed["Distance"],
            boxcox_store["distance_lambda"],
            boxcox_store.get("distance_offset", 0),
        )

    if "Rooms" in processed.columns:
        processed["Rooms_Transformed"] = box_cox_transform(
            processed["Rooms"],
            boxcox_store["rooms_lambda"],
            boxcox_store.get("rooms_offset", 0),
        )

    if "Bathroom" in processed.columns:
        processed["Bathroom_Transformed"] = box_cox_transform(
            processed["Bathroom"],
            boxcox_store["bathroom_lambda"],
            boxcox_store.get("bathroom_offset", 0),
        )

    if "Car" in processed.columns:
        processed["Car_Transformed"] = box_cox_transform(
            processed["Car"],
            boxcox_store["car_lambda"],
            boxcox_store.get("car_offset", 0),
        )

    if "PropertyAge" in processed.columns:
        processed["PropertyAge_Transformed"] = box_cox_transform(
            processed["PropertyAge"],
            boxcox_store["propertyage_lambda"],
            boxcox_store.get("propertyage_offset", 0),
        )

    # Convert boolean values to integers
    bool_columns = processed.select_dtypes(include=["bool"]).columns
    for col in bool_columns:
        processed[col] = processed[col].astype(int)

    # 7. Add any missing columns required by the model with default values (0)
    required_columns = [
        "PropType_House",
        "PropType_Townhouse",
        "PropType_Unit/Apartment",
        "Method_PI",
        "Method_S",
        "Method_SA",
        "Method_SP",
        "Method_VB",
        "Suburb_PriceRank",
        "Seller_Barry",
        "Seller_Biggin",
        "Seller_Brad",
        "Seller_Buxton",
        "Seller_Fletchers",
        "Seller_Gary",
        "Seller_Greg",
        "Seller_Harcourts",
        "Seller_Hodges",
        "Seller_Jas",
        "Seller_Jellis",
        "Seller_Kay",
        "Seller_Love",
        "Seller_Marshall",
        "Seller_McGrath",
        "Seller_Miles",
        "Seller_Nelson",
        "Seller_Noel",
        "Seller_Other",
        "Seller_RT",
        "Seller_Raine",
        "Seller_Ray",
        "Seller_Stockdale",
        "Seller_Sweeney",
        "Seller_Village",
        "Seller_Williams",
        "Seller_Woodards",
        "Seller_YPA",
        "Seller_hockingstuart",
        "Landsize_Transformed",
        "BuildingArea_Transformed",
        "Distance_Transformed",
        "Rooms_Transformed",
        "Bathroom_Transformed",
        "Car_Transformed",
        "PropertyAge_Transformed",
    ]

    for col in required_columns:
        if col not in processed.columns:
            processed[col] = 0

    return processed


def box_cox_transform(values, lambda_val, offset=0):
    """Apply Box-Cox transformation to a series of values"""
    values = pd.to_numeric(values, errors="coerce")
    # Handle NaNs
    values = values.fillna(values.median())

    # Apply offset if needed
    values_offset = values + offset

    # Apply transformation
    if abs(lambda_val) < 1e-10:  # lambda is close to zero
        return np.log(values_offset)
    else:
        return ((values_offset**lambda_val) - 1) / lambda_val


def inverse_transform_price(transformed_value):
    """
    Inverse-transform a price value using the stored PowerTransformer.

    Args:
        transformed_value: Transformed price value

    Returns:
        Original price value
    """
    # Get the offset
    offset = boxcox_store.get("price_offset", 0)

    # Get the transformer from boxcox_store
    if "price_transformer" in boxcox_store:
        pt = boxcox_store["price_transformer"]

        # Reshape for scikit-learn
        value_reshaped = np.array([transformed_value]).reshape(-1, 1)

        # Use the transformer's built-in inverse_transform method
        original_with_offset = pt.inverse_transform(value_reshaped)[0][0]

        # Remove the offset
        original_price = original_with_offset - offset

        return original_price
    else:
        # Fallback to manual implementation if transformer isn't available
        print("Warning: PowerTransformer not found, using fallback method")
        lambda_val = boxcox_store["price_lambda"]

        if abs(lambda_val) < 1e-10:  # lambda is close to zero
            x_original = np.exp(transformed_value)
        else:
            x_original = np.power(lambda_val * transformed_value + 1, 1 / lambda_val)

        return x_original - offset


input_data = (
    {
        "Suburb": "Reservoir",
        "Rooms": 3,
        "Type": "House",
        "Method": "S",
        "Seller": "Ray",
        "Distance": 11.2,
        "Bathroom": 1.0,
        "Car": 2,
        "Landsize": 556.0,
        "BuildingArea": 120.0,
        "PropertyAge": 50,
        "Direction": "N",
        "LandSizeNotOwned": False,
    },
)

predict(input_data)

"""
custom
'before processing'

{'Suburb': {0: 'Reservoir'},
 'Rooms': {0: 3},
 'Type': {0: 'House'},
 'Method': {0: 'S'},
 'Seller': {0: 'Ray'},
 'Distance': {0: 11.2},
 'Bathroom': {0: 1.0},
 'Car': {0: 2},
 'Landsize': {0: 556.0},
 'BuildingArea': {0: 120.0},
 'PropertyAge': {0: 50},
 'Direction': {0: 'N'},
 'LandSizeNotOwned': {0: False}}

'after processing'

{'Rooms': {0: 3},
 'Distance': {0: 11.2},
 'Bathroom': {0: 1.0},
 'Car': {0: 2},
 'Landsize': {0: 556.0},
 'BuildingArea': {0: 120.0},
 'PropertyAge': {0: 50},
 'LandSizeNotOwned': {0: 0},
 'PropType_House': {0: 1},
 'PropType_Townhouse': {0: 0},
 'PropType_Unit/Apartment': {0: 0},
 'Method_PI': {0: 0},
 'Method_S': {0: 1},
 'Method_SA': {0: 0},
 'Method_SP': {0: 0},
 'Method_VB': {0: 0},
 'Suburb_PriceRank': {0: 0.30670926517571884},
 'Seller_Barry': {0: 0},
 'Seller_Biggin': {0: 0},
 'Seller_Brad': {0: 0},
 'Seller_Buxton': {0: 0},
 'Seller_Fletchers': {0: 0},
 'Seller_Gary': {0: 0},
 'Seller_Greg': {0: 0},
 'Seller_Harcourts': {0: 0},
 'Seller_Hodges': {0: 0},
 'Seller_Jas': {0: 0},
 'Seller_Jellis': {0: 0},
 'Seller_Kay': {0: 0},
 'Seller_Love': {0: 0},
 'Seller_Marshall': {0: 0},
 'Seller_McGrath': {0: 0},
 'Seller_Miles': {0: 0},
 'Seller_Nelson': {0: 0},
 'Seller_Noel': {0: 0},
 'Seller_RT': {0: 0},
 'Seller_Raine': {0: 0},
 'Seller_Ray': {0: 1},
 'Seller_Stockdale': {0: 0},
 'Seller_Sweeney': {0: 0},
 'Seller_Village': {0: 0},
 'Seller_Williams': {0: 0},
 'Seller_Woodards': {0: 0},
 'Seller_YPA': {0: 0},
 'Seller_hockingstuart': {0: 0},
 'Direction_N': {0: 1},
 'Direction_S': {0: 0},
 'Direction_E': {0: 0},
 'Direction_W': {0: 0},
 'Landsize_Transformed': {0: 4.748195467838697},
 'BuildingArea_Transformed': {0: 18.751731150684726},
 'Distance_Transformed': {0: 3.5750088276141323},
 'Rooms_Transformed': {0: 1.6697705969946388},
 'Bathroom_Transformed': {0: -0.0},
 'Car_Transformed': {0: 1.298438643090085},
 'PropertyAge_Transformed': {0: 15.802135704940705},
 'Seller_Other': {0: 0}}
"""


"""
bentoml
before processing {'Suburb': {0: 'Reservoir'}, 'Rooms': {0: 3}, 'Type': {0: 'House'}, 'Method': {0: 'S'}, 'Seller': {0: 'Ray'}, 'Distance': {0: 11.2}, 'Bathroom': {0: 1.0}, 'Car': {0: 2}, 'Landsize': {0: 556.0}, 'BuildingArea': {0: 120.0}, 'PropertyAge': {0: 50}, 'Direction': {0: 'N'}, 'LandSizeNotOwned': {0: False}}
2025-02-27T08:28:26Z[Service: housin...iction][Replica: rrl64]after processing {'Rooms': {0: 3}, 'Distance': {0: 11.2}, 'Bathroom': {0: 1.0}, 'Car': {0: 2}, 'Landsize': {0: 556.0}, 'BuildingArea': {0: 120.0}, 'PropertyAge': {0: 50}, 'LandSizeNotOwned': {0: 0}, 'PropType_House': {0: 1}, 'PropType_Townhouse': {0: 0}, 'PropType_Unit/Apartment': {0: 0}, 'Method_PI': {0: 0}, 'Method_S': {0: 1}, 'Method_SA': {0: 0}, 'Method_SP': {0: 0}, 'Method_VB': {0: 0}, 'Suburb_PriceRank': {0: nan}, 'Seller_Barry': {0: 0}, 'Seller_Biggin': {0: 0}, 'Seller_Brad': {0: 0}, 'Seller_Buxton': {0: 0}, 'Seller_Fletchers': {0: 0}, 'Seller_Gary': {0: 0}, 'Seller_Greg': {0: 0}, 'Seller_Harcourts': {0: 0}, 'Seller_Hodges': {0: 0}, 'Seller_Jas': {0: 0}, 'Seller_Jellis': {0: 0}, 'Seller_Kay': {0: 0}, 'Seller_Love': {0: 0}, 'Seller_Marshall': {0: 0}, 'Seller_McGrath': {0: 0}, 'Seller_Miles': {0: 0}, 'Seller_Nelson': {0: 0}, 'Seller_Noel': {0: 0}, 'Seller_RT': {0: 0}, 'Seller_Raine': {0: 0}, 'Seller_Ray': {0: 1}, 'Seller_Stockdale': {0: 0}, 'Seller_Sweeney': {0: 0}, 'Seller_Village': {0: 0}, 'Seller_Williams': {0: 0}, 'Seller_Woodards': {0: 0}, 'Seller_YPA': {0: 0}, 'Seller_hockingstuart': {0: 0}, 'Seller_Other': {0: 0}, 'Direction_N': {0: 1}, 'Direction_S': {0: 0}, 'Direction_E': {0: 0}, 'Direction_W': {0: 0}, 'Landsize_Transformed': {0: 4.748195467838697}, 'BuildingArea_Transformed': {0: 18.751731150684726}, 'Distance_Transformed': {0: 3.5750088276141323}, 'Rooms_Transformed': {0: 1.6697705969946388}, 'Bathroom_Transformed': {0: -0.0}, 'Car_Transformed': {0: 1.2984386430900843}, 'PropertyAge_Transformed': {0: 15.802135704940705}}
"""
