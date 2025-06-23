from google.cloud import aiplatform
from typing import Dict, Union
import tensorflow as tf
import numpy as np


def preprocess(instance: Dict) -> Dict:
    """Preprocess the input instance for Vertex AI"""
    input_data = {
        "job_description_input": tf.reshape(
            tf.convert_to_tensor([instance["job_description"]], dtype=tf.string),
            (-1, 1),
        ),
        "job_title_input": tf.reshape(
            tf.convert_to_tensor([instance["job_title"]], dtype=tf.string), (-1, 1)
        ),
        "query_input": tf.reshape(
            tf.convert_to_tensor([instance["query"]], dtype=tf.string), (-1, 1)
        ),
        "soft_skills_input": tf.reshape(
            tf.convert_to_tensor(
                [
                    " ".join(instance["soft_skills"])
                    if isinstance(instance["soft_skills"], list)
                    else instance["soft_skills"]
                ],
                dtype=tf.string,
            ),
            (-1, 1),
        ),
        "hard_skills_input": tf.reshape(
            tf.convert_to_tensor(
                [
                    " ".join(instance["hard_skills"])
                    if isinstance(instance["hard_skills"], list)
                    else instance["hard_skills"]
                ],
                dtype=tf.string,
            ),
            (-1, 1),
        ),
        "location_flexibility_input": tf.reshape(
            tf.convert_to_tensor([instance["location_flexibility"]], dtype=tf.string),
            (-1, 1),
        ),
        "contract_type_input": tf.reshape(
            tf.convert_to_tensor([instance["contract_type"]], dtype=tf.string), (-1, 1)
        ),
        "education_level_input": tf.reshape(
            tf.convert_to_tensor([instance["education_level"]], dtype=tf.string),
            (-1, 1),
        ),
        "seniority_input": tf.reshape(
            tf.convert_to_tensor([instance["seniority"]], dtype=tf.string), (-1, 1)
        ),
        "min_years_experience_input": tf.reshape(
            tf.convert_to_tensor(
                [float(instance["min_years_experience"])], dtype=tf.float32
            ),
            (-1, 1),
        ),
        "field_of_study_input": tf.reshape(
            tf.convert_to_tensor(
                [
                    " ".join(instance["field_of_study"])
                    if isinstance(instance["field_of_study"], list)
                    else instance["field_of_study"]
                ],
                dtype=tf.string,
            ),
            (-1, 1),
        ),
    }
    return input_data


def postprocess(prediction: Dict, country_code: str = "US") -> Dict:
    """Postprocess the prediction output"""
    output_key = list(prediction.keys())[0]
    salary_prediction = prediction[output_key].numpy()

    if country_code == "IN":
        final_salary = float(np.expm1(np.expm1(salary_prediction)))
    else:
        final_salary = float(np.expm1(salary_prediction))

    return {"predicted_salary": final_salary}


# Initialize Vertex AI
aiplatform.init(
    project="your-project-id",
    location="your-region",  # e.g., 'us-central1'
)

# Deploy the model
model = aiplatform.Model.upload(
    display_name="salary-prediction-model",
    artifact_uri="gs://your-bucket/path/to/saved_model",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
)

# Create an endpoint
endpoint = aiplatform.Endpoint.create(display_name="salary-prediction-endpoint")

# Deploy model to endpoint
endpoint.deploy(
    model=model, machine_type="n1-standard-2", min_replica_count=1, max_replica_count=1
)


def predict_salary_vertex(
    project: str, endpoint_id: str, location: str, instance: Dict
) -> float:
    """
    Make prediction using Vertex AI endpoint
    """
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    )

    prediction = endpoint.predict(instances=[instance])
    return prediction.predictions[0]


# Example usage:
instance = {
    "job_description": "Software engineer position...",
    "job_title": "Software Engineer",
    "query": "software engineer",
    "soft_skills": ["communication", "teamwork"],
    "hard_skills": ["python", "javascript"],
    "location_flexibility": "remote",
    "contract_type": "full_time",
    "education_level": "bachelor",
    "seniority": "mid",
    "min_years_experience": 3,
    "field_of_study": ["computer science"],
    "country_code": "SG",
}

predicted_salary = predict_salary_vertex(
    project="your-project-id",
    endpoint_id="your-endpoint-id",
    location="your-region",
    instance=instance,
)

print(f"Predicted salary: ${predicted_salary:,.2f}")


def predict_salary_vertex_with_retry(
    project: str, endpoint_id: str, location: str, instance: Dict, max_retries: int = 3
) -> Union[float, None]:
    """
    Make prediction using Vertex AI endpoint with retry logic
    """
    from google.api_core import retry

    @retry.Retry(
        predicate=retry.if_transient_error, initial=1.0, maximum=10.0, multiplier=2.0
    )
    def _do_predict():
        try:
            endpoint = aiplatform.Endpoint(
                endpoint_name=f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
            )
            prediction = endpoint.predict(instances=[instance])
            return prediction.predictions[0]
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

    for attempt in range(max_retries):
        try:
            return _do_predict()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None
            continue
