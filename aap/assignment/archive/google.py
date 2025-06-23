import pandas as pd
import google.generativeai as genai
import os
from typing import Dict, List, Optional
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class JobFeatureExtractor:
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

    def extract_features(self, description: str) -> Dict:
        """Extract features from job description using Gemini"""
        import time

        # Default empty response structure
        default_response = {
            "soft_skills": [],
            "hard_skills": [],
            "location_flexibility": "unspecified",
            "contract_type": "unspecified",
            "education_level": "unspecified",
            "field_of_study": "unspecified",
            "min_years_experience": -1,
            "salary_range": {
                "min": -1,
                "max": -1,
                "currency": "unspecified",
                "period": "unspecified",
            },
        }

        prompt = f"""You are a JSON generator. Your task is to analyze this job posting and return ONLY a valid JSON object with no additional text or formatting. Extract the following features:
        - soft_skills: List of soft skills mentioned (communication, leadership, etc)
        - hard_skills: List of technical skills, tools, languages required
        - location_flexibility: One of [remote, hybrid, onsite, unspecified]
        - contract_type: One of [full-time, part-time, contract, internship, unspecified] 
        - education_level: Minimum required education level [high_school, bachelors, masters, phd, unspecified]
        - field_of_study: Required field of study or major
        - min_years_experience: Minimum years of experience required (numeric, -1 if unspecified)
        - salary_range: Extract salary range if available [min, max, currency, period(yearly/monthly/hourly)]
        
        Job Details:
        {description}

        IMPORTANT: Return ONLY a valid JSON object. No other text, no markdown formatting, no explanations.
        Example format:
        {{"soft_skills": ["communication"], "hard_skills": ["python"], "location_flexibility": "remote", "contract_type": "full-time", "education_level": "bachelors", "field_of_study": "computer science", "min_years_experience": 3, "salary_range": {{"min": 80000, "max": 100000, "currency": "USD", "period": "yearly"}}}}
        """

        max_retries = 3
        current_retry = 0

        try:
            while current_retry < max_retries:
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0,
                            top_p=1,
                            top_k=1,
                            max_output_tokens=1024,
                        ),
                    )
                    break  # If successful, break out of the retry loop

                except Exception as e:
                    if "429" in str(e):  # Rate limit error
                        current_retry += 1
                        if current_retry < max_retries:
                            print(
                                f"\nRate limit hit. Sleeping for 5 seconds... (Attempt {current_retry}/{max_retries})"
                            )
                            time.sleep(5)
                            continue
                    raise  # Re-raise the exception if we've exhausted retries or it's a different error

            # Clean and extract JSON from response
            response_text = response.text.strip()

            # Try to find JSON content if wrapped in other text
            try:
                # First attempt: direct JSON parse
                features = json.loads(response_text)
            except json.JSONDecodeError:
                # Second attempt: try to find JSON-like structure
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx : end_idx + 1]
                    try:
                        features = json.loads(json_str)
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON: {json_str}")
                        return default_response
                else:
                    print("Could not find valid JSON in response")
                    return default_response

            # Ensure all required fields are present
            for key in default_response.keys():
                if key not in features:
                    features[key] = default_response[key]

            return features

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return default_response


def main():
    # Read both CSV files
    glassdoor_df = pd.read_csv("glassdoor.csv")

    try:
        existing_eda_df = pd.read_csv("eda.csv")
        # Get the number of rows already processed
        processed_rows = len(existing_eda_df)
        print(f"Found {processed_rows} existing processed rows in eda.csv")

        # Get the remaining rows from glassdoor.csv
        df = glassdoor_df.iloc[processed_rows:]

        if len(df) == 0:
            print("All rows have been processed already!")
            return

    except FileNotFoundError:
        print("No existing eda.csv found. Creating new eda.csv file...")
        df = glassdoor_df
        existing_eda_df = None
        # Create empty eda.csv with headers
        empty_df = pd.DataFrame(
            columns=[
                "job_title",
                "company_name",
                "location",
                "salary",
                "job_description",
                "soft_skills",
                "hard_skills",
                "location_flexibility",
                "contract_type",
                "education_level",
                "field_of_study",
                "min_years_experience",
                "salary_range",
            ]
        )
        empty_df.to_csv("eda.csv", index=False)
        print("Created empty eda.csv with headers")

    # Initialize feature extractor
    extractor = JobFeatureExtractor()

    # Reset index of remaining rows to process
    df = df.reset_index(drop=True)

    # Process each job and append immediately to CSV
    total_rows = len(df)
    for idx, row in tqdm(df.iterrows(), desc="Extracting features", total=total_rows):
        prompt_text = f"""
        Job Title: {row['job_title']}
        Location: {row['location']}
        Salary: {row['salary']}
        Description: {row['job_description']}
        """

        # Extract features for current job
        features = extractor.extract_features(prompt_text)

        # Create a new DataFrame with both the original row and features
        combined_data = {**row.to_dict(), **features}
        current_result = pd.DataFrame([combined_data])

        # Append mode if not first row, write mode if first row
        mode = "a" if idx > 0 or existing_eda_df is not None else "w"
        header = not (mode == "a")  # Only write header for first row

        # Append to CSV immediately
        current_result.to_csv("eda.csv", mode=mode, header=header, index=False)


main()
