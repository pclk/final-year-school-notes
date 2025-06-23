import pandas as pd
import os
from typing import Dict, List, Optional
import json
from tqdm import tqdm
import anthropic
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env with poetry self add poetry-plugin-dotenv


class JobFeatureExtractor:
    def __init__(self):
        self.client = anthropic.Anthropic()

    async def extract_features_batch(self, descriptions: List[Dict]):
        """Extract features from multiple job descriptions using Claude"""
        # Prepare batch requests
        requests = []
        for idx, desc in enumerate(descriptions):
            prompt = f"""You are a JSON generator. Your task is to analyze this job posting and return ONLY a valid JSON object with no additional text or formatting. Here is the job posting:

Job Title: {desc['job_title']}
Location: {desc['location']}
Salary: {desc['salary']}
Country: {desc['country']}
Description: {desc['job_description']}

Extract and return ONLY this JSON structure:

{{
    "soft_skills": ["list", "of", "soft skills"],  // Communication, leadership, teamwork, problem-solving, etc.
    "hard_skills": ["list", "of", "technical skills"],  // Programming languages, tools, frameworks, certifications
    "location_flexibility": "enum",  // Exactly one of: ["remote", "hybrid", "onsite", "unspecified"]
    "contract_type": "enum",  // Exactly one of: ["full-time", "part-time", "contract", "internship", "unspecified"]
    "education_level": "enum",  // Exactly one of: ["high_school", "bachelors", "masters", "phd", "unspecified"]
    "field_of_study": ["list", "of", "field of studies"], // for example: ["computer_science", "software_engineering"]
    "seniority": "enum", // Exactly one of ["junior", "mid-level", "senior", "unspecified"]
    "min_years_experience": number,  // Minimum years required, use -1 if unspecified
    "min_salary": number,  // Minimum salary amount, use -1 if unspecified
    "max_salary": number,  // Maximum salary amount, use -1 if unspecified
    "salary_period": "enum"  // Exactly one of: ["yearly", "monthly", "hourly", "unspecified"]
}}

Rules:
1. Return ONLY valid JSON, no explanations or additional text
2. Use exact enum values as specified
3. Use -1 for any numeric fields that cannot be determined
4. Use "unspecified" for any string/enum fields that cannot be determined
5. Always include all fields in the response
6. Lists should be empty [] if no values found
7. Normalize education levels (e.g., "BS", "Bachelor's", "Bachelors" all become "bachelors")
8. Convert all salary amounts to numbers (no currency symbols or commas)
9. Use the provided salary information to determine salary details when available"""

            requests.append(
                anthropic.types.messages.batch_create_params.Request(  # type: ignore
                    custom_id=f"job_{idx}",
                    params=anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                        model="claude-3-5-haiku-latest",
                        max_tokens=1024,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
            )

        try:
            # Print request details
            print("\nSending batch request to Anthropic:")
            print(f"Number of messages in batch: {len(requests)}")

            # Create batch request and store the response
            message_batch = await asyncio.get_event_loop().run_in_executor(
                ThreadPoolExecutor(),
                lambda: self.client.messages.batches.create(requests=requests),
            )
            self.message_batch = message_batch  # Store for later reference

            # Print response details
            print("\nReceived response from Anthropic:")
            print(f"Batch ID: {message_batch.id}")
            print(f"Created at: {message_batch.created_at}")
            print(f"Expires at: {message_batch.expires_at}")

            # Save the complete message_batch response
            # Convert datetime objects to ISO format strings
            batch_info = {
                "id": message_batch.id,
                "type": message_batch.type,
                "processing_status": message_batch.processing_status,
                "request_counts": {
                    "processing": message_batch.request_counts.processing,
                    "succeeded": message_batch.request_counts.succeeded,
                    "errored": message_batch.request_counts.errored,
                    "canceled": message_batch.request_counts.canceled,
                    "expired": message_batch.request_counts.expired,
                },
                "ended_at": message_batch.ended_at.isoformat()
                if message_batch.ended_at
                else None,
                "created_at": message_batch.created_at.isoformat()
                if message_batch.created_at
                else None,
                "expires_at": message_batch.expires_at.isoformat()
                if message_batch.expires_at
                else None,
                "cancel_initiated_at": message_batch.cancel_initiated_at.isoformat()
                if message_batch.cancel_initiated_at
                else None,
                "results_url": message_batch.results_url,
            }

            with open("batch_tracking.json", "w") as f:
                json.dump(batch_info, f, indent=2)

            print("\nBatch tracking information saved to batch_tracking.json")
            print("Batch successfully submitted! Exiting without polling.")

            return

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return


async def main():
    # Read cleaned CSV file
    print("Loading cleaned Glassdoor data...")
    glassdoor_df = pd.read_csv("cleaned_glassdoor.csv")
    print(f"Loaded {len(glassdoor_df)} cleaned job listings")

    try:
        # Try to load existing progress
        processed_files = [
            f
            for f in os.listdir("results")
            if f.startswith("batch_") and f.endswith(".csv")
        ]
        if processed_files:
            processed_rows = sum(
                [len(pd.read_csv(f"results/{f}")) for f in processed_files]
            )
            print(
                f"Found {processed_rows} processed rows across {len(processed_files)} batch files"
            )
            df = glassdoor_df.iloc[processed_rows:]
        else:
            print("No existing processed batches found. Starting from beginning...")
            df = glassdoor_df
            processed_rows = 0

        if len(df) == 0:
            print("All rows have been processed! Combining results...")
            # Combine all batch files into final result
            all_batches = [pd.read_csv(f"results/{f}") for f in processed_files]
            final_df = pd.concat(all_batches, ignore_index=True)
            final_df.to_csv("anthropic_features.csv", index=False)
            print(
                f"Successfully combined all batches into anthropic_features.csv with {len(final_df)} rows"
            )
            return

    except FileNotFoundError:
        print("No existing anthropic.csv found. Creating new anthropic.csv file...")
        df = glassdoor_df
        existing_eda_df = None
        empty_df = pd.DataFrame(
            columns=[
                "query",
                "country",
                "job_description",
                "location",
                "salary",
                "job_title",
                "job_link",
                "soft_skills",
                "hard_skills",
                "location_flexibility",
                "contract_type",
                "education_level",
                "field_of_study",
                "seniority",
                "min_years_experience",
                "min_salary",
                "max_salary",
                "salary_period",
            ]  # type: ignore
        )
        empty_df.to_csv("anthropic.csv", index=False)
        print("Created empty anthropic.csv with headers")

    # Initialize feature extractor
    extractor = JobFeatureExtractor()

    # Reset index of remaining rows to process
    df = df.reset_index(drop=True)

    # Print total number of jobs to process
    total_jobs = len(df)
    print(f"\nProcessing all {total_jobs} jobs in a single batch request")

    # Prepare all prompts
    prompts = [row.to_dict() for _, row in df.iterrows()]

    # Extract features for all jobs
    print("Submitting batch request...")
    await extractor.extract_features_batch(prompts)

    print(f"Batch submitted! ID: {extractor.message_batch.id}")
    print("You can now use poll_batch.py to check status and retrieve results")


if __name__ == "__main__":
    asyncio.run(main())
