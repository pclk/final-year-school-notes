import anthropic
import json
import time
import pandas as pd
import os
from datetime import datetime


def load_batch_info():
    with open("batch_tracking.json", "r") as f:
        return json.load(f)


# Remove this function as we're now handling the save directly in poll_batch


def poll_batch():
    client = anthropic.Anthropic()
    batch_info = load_batch_info()

    if batch_info["processing_status"] == "ended":
        print("Batch already completed!")
        return

    print(f"Polling batch {batch_info['id']}...")

    while True:
        message_batch = client.messages.batches.retrieve(batch_info["id"])

        # Print current status
        print(f"\nStatus: {message_batch.processing_status}")
        print(f"Processing: {message_batch.request_counts.processing}")
        print(f"Succeeded: {message_batch.request_counts.succeeded}")
        print(f"Errored: {message_batch.request_counts.errored}")
        print(f"Canceled: {message_batch.request_counts.canceled}")
        print(f"Expired: {message_batch.request_counts.expired}")

        if message_batch.ended_at:
            print(f"Ended at: {message_batch.ended_at}")
        print(f"Created at: {message_batch.created_at}")
        print(f"Expires at: {message_batch.expires_at}")

        if message_batch.results_url:
            print(f"Results URL: {message_batch.results_url}")

        # Save the latest status
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

        if message_batch.processing_status == "ended":
            print("\nBatch processing complete!")
            return True

        time.sleep(10)


def list_batches():
    client = anthropic.Anthropic()
    print("\nListing all message batches...")

    try:
        batches = client.messages.batches.list()

        print("\nCompleted Batches:")
        print("-----------------")
        for batch in batches.data:
            print(f"\nBatch ID: {batch.id}")
            print(f"Status: {batch.processing_status}")
            print(f"Created: {batch.created_at}")
            print(f"Requests - Total: {sum(batch.request_counts.__dict__.values())}")
            print(f"         - Succeeded: {batch.request_counts.succeeded}")
            print(f"         - Failed: {batch.request_counts.errored}")
            print(
                f"Results URL: {batch.results_url if batch.results_url else 'Not available'}"
            )
            print("-----------------")

    except Exception as e:
        print(f"Error listing batches: {str(e)}")


def retrieve_results():
    client = anthropic.Anthropic()
    batch_info = load_batch_info()

    if batch_info["processing_status"] != "ended":
        print("Batch is not yet complete!")
        return

    if not batch_info.get("results_url"):
        print("No results URL available!")
        return

    print(f"Retrieving results for batch {batch_info['id']}...")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Process results
    all_results = []
    success_count = 0
    error_count = 0

    for result in client.messages.batches.results(batch_info["id"]):
        if result.result.type == "succeeded":
            try:
                response_text = result.result.message.content[0].text.strip()
                json_data = json.loads(response_text)
                all_results.append(
                    {
                        "custom_id": result.custom_id,
                        "status": "success",
                        "data": json_data,
                    }
                )
                success_count += 1
            except Exception as e:
                error_count += 1
                all_results.append(
                    {"custom_id": result.custom_id, "status": "error", "error": str(e)}
                )
        else:
            error_count += 1
            all_results.append(
                {
                    "custom_id": result.custom_id,
                    "status": "error",
                    "error": result.result.type,
                }
            )

        # Save progress every 1000 results
        if len(all_results) % 1000 == 0:
            print(f"Processed {len(all_results)} results...")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/batch_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print(f"Successful results: {success_count}")
    print(f"Failed results: {error_count}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        [
            {
                "job_id": result["custom_id"],
                "soft_skills": ",".join(result["data"]["soft_skills"])
                if result["status"] == "success"
                else "",
                "hard_skills": ",".join(result["data"]["hard_skills"])
                if result["status"] == "success"
                else "",
                "location_flexibility": result["data"]["location_flexibility"]
                if result["status"] == "success"
                else "",
                "contract_type": result["data"]["contract_type"]
                if result["status"] == "success"
                else "",
                "education_level": result["data"]["education_level"]
                if result["status"] == "success"
                else "",
                "field_of_study": ",".join(result["data"]["field_of_study"])
                if result["status"] == "success"
                else "",
                "seniority": result["data"]["seniority"]
                if result["status"] == "success"
                else "",
                "min_years_experience": result["data"]["min_years_experience"]
                if result["status"] == "success"
                else None,
                "min_salary": result["data"]["min_salary"]
                if result["status"] == "success"
                else None,
                "max_salary": result["data"]["max_salary"]
                if result["status"] == "success"
                else None,
                "salary_period": result["data"]["salary_period"]
                if result["status"] == "success"
                else "",
            }
            for result in all_results
        ]
    )

    # Read the cleaned Glassdoor data
    glassdoor_df = pd.read_csv("cleaned_glassdoor.csv")

    # Add job_id column to match with results
    glassdoor_df["job_id"] = glassdoor_df.index.map(lambda x: f"job_{x}")

    # Merge the dataframes
    merged_df = pd.merge(glassdoor_df, results_df, on="job_id", how="left")

    # Save the merged dataset
    merged_df.to_csv("jobs.csv", index=False)
    print("\nMerged results saved to jobs.csv")
    print(f"Total rows in final dataset: {len(merged_df)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["poll", "retrieve", "list"], help="Action to perform"
    )
    args = parser.parse_args()

    if args.action == "poll":
        poll_batch()
    elif args.action == "list":
        list_batches()
    else:
        retrieve_results()
