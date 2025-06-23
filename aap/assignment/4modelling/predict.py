import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
import joblib
import os
import warnings
import tensorflow as tf

# Disable all warnings
warnings.filterwarnings("ignore")

# Disable tensorflow warnings
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def preprocess_text(text):
    if text is None:
        return []
    words = str(text).lower().split()
    words = [w.strip(".,!?()[]{}:;") for w in words if len(w) > 2]
    return words


def create_sequence_vector(sequence, w2v_model):
    vectors = [w2v_model.wv[token] for token in sequence if token in w2v_model.wv]
    if not vectors:
        return np.zeros(100)  # WORD2VEC_SIZE = 100
    return np.mean(vectors, axis=0)


def predict_salary(
    country_code,
    job_title,
    job_description,
    location,
    min_years_experience,
    location_flexibility,
    contract_type,
    education_level,
    seniority,
    query="",
    soft_skills=[],
    hard_skills=[],
    field_of_study=[],
):
    """
    Predict salary based on job details

    Parameters:
    - country_code: str ("US", "IN", or "SG")
    - job_title: str
    - job_description: str
    - location: str
    - min_years_experience: float
    - location_flexibility: str
    - contract_type: str
    - education_level: str
    - seniority: str
    - query: str (optional)
    - soft_skills: list of str (optional)
    - hard_skills: list of str (optional)
    - field_of_study: list of str (optional)

    Returns:
    - predicted_salary: float
    """

    # Check if country model exists
    model_dir = f"country_models/{country_code}"
    if not os.path.exists(model_dir):
        raise ValueError(f"No model found for country: {country_code}")

    # Load models
    w2v_model = Word2Vec.load(f"{model_dir}/word2vec_model.bin")
    deep_model = load_model(
        f"{model_dir}/deep_model.h5", custom_objects={"mse": "mean_squared_error"}
    )
    scaler = joblib.load(f"{model_dir}/scaler.joblib")

    # Create sequence
    sequence = []

    # Add skills and fields
    sequence.extend([f"soft_{skill.lower()}" for skill in soft_skills])
    sequence.extend([f"hard_{skill.lower()}" for skill in hard_skills])
    sequence.extend([f"field_{field.lower()}" for field in field_of_study])

    # Process text fields
    sequence.extend([f"desc_{word}" for word in preprocess_text(job_description)])
    sequence.extend([f"title_{word}" for word in preprocess_text(job_title)])
    sequence.extend([f"query_{word}" for word in preprocess_text(query)])

    if location:
        sequence.append(f"loc_{str(location).lower()}")

    # Add categorical features
    for col, value in {
        "location_flexibility": location_flexibility,
        "contract_type": contract_type,
        "education_level": education_level,
        "seniority": seniority,
    }.items():
        if value:
            sequence.append(f"{col}_{str(value).lower()}")

    # Create feature vector
    X = create_sequence_vector(sequence, w2v_model)

    # Add numerical features
    X_numeric = scaler.transform([[min_years_experience]])
    X = np.hstack((X.reshape(1, -1), X_numeric))

    # Make prediction
    prediction = deep_model.predict(X, verbose=0)[0][0]

    # Transform prediction back to original scale
    if country_code == "IN":
        # Double inverse transform for India
        predicted_salary = np.expm1(np.expm1(prediction))
    else:
        # Single inverse transform for other countries
        predicted_salary = np.expm1(prediction)

    return predicted_salary


# Example usage:
if __name__ == "__main__":
    # Example prediction for each country
    cities = {"US": "New York", "IN": "Bangalore", "SG": "Singapore"}

    for country_code, city in cities.items():
        try:
            salary = predict_salary(
                country_code=country_code,
                job_title="Senior Software Engineer",
                job_description="Looking for an experienced developer with Python and ML skills",
                location=city,
                min_years_experience=5,
                location_flexibility="remote",
                contract_type="full_time",
                education_level="bachelors",
                seniority="senior",
                soft_skills=["communication", "leadership"],
                hard_skills=["python", "machine learning"],
                field_of_study=["computer science"],
            )
            print(f"{country_code} Predicted salary: ${salary:,.2f} USD per year")
        except Exception as e:
            print(f"Error making prediction for {country_code}: {str(e)}")
