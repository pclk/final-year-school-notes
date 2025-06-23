import pandas as pd
from anthropic import Anthropic
import tiktoken
import json
from typing import Dict, Tuple, Set, Union, Optional
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from tqdm import tqdm

import numpy as np
from rapidfuzz import fuzz  # pip install rapidfuzz
from fast_langdetect import detect  # pip install fast-langdetect
import re
import nltk
from nltk.corpus import stopwords

# Constants
SIMILARITY_THRESHOLD = 90  # (%)
MIN_DESCRIPTION_LENGTH = 300  # Characters
INVALID_TITLE = "Title not available"  # Invalid title string
MIN_TEXT_LENGTH_FOR_LANG_CHECK = 10  # Characters
MIN_LANG_CONFIDENCE_SCORE = 0.1  # 0-1 float probablity
LENGTH_RATIO_THRESHOLD = 0.7  # 0-1 float probablity
FINGERPRINT_LENGTH = 100  # Characters
HASH_PREFIX_LENGTH = 8  # Characters

# Claude API pricing constants
CLAUDE_INPUT_COST_PER_1M = 0.4
CLAUDE_OUTPUT_COST_PER_1M = 2.0

# Required Columns
REQUIRED_COLUMNS = [
    "query",
    "country",
    "job_description",
    "location",
    "salary",
    "job_title",
    "job_link",
]

# File Paths
INPUT_FILE = "glassdoor.csv"
CLEANED_OUTPUT_FILE = "cleaned_glassdoor.csv"
SIMILAR_DESC_FILE = "similar_descriptions_analysis.csv"
PROBLEMATIC_ENTRIES_FILE = "problematic_entries.csv"

TEMPLATE = """You are a JSON generator. Your task is to analyze this job posting and return ONLY a valid JSON object with no additional text or formatting. Extract the following features:

{
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
    "salary_currency": "string",  // USD, EUR, etc. Use "unspecified" if unclear
    "salary_period": "enum"  // Exactly one of: ["yearly", "monthly", "hourly", "unspecified"]
}

Rules:
1. Return ONLY valid JSON, no explanations or additional text
2. Use exact enum values as specified
3. Use -1 for any numeric fields that cannot be determined
4. Use "unspecified" for any string/enum fields that cannot be determined
5. Always include all fields in the response
6. Lists should be empty [] if no values found
7. Normalize education levels (e.g., "BS", "Bachelor's", "Bachelors" all become "bachelors")
8. Convert all salary amounts to numbers (no currency symbols or commas)"""


# Initialize NLTK resources and patterns
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
WORD_PATTERN = re.compile(r"\b\w+\b")


def find_similar_descriptions(
    df: pd.DataFrame, threshold: float = 85
) -> Tuple[set, list]:
    """Find similar job descriptions using rapid fuzzy matching

    Args:
        df: DataFrame containing job descriptions
        threshold: Similarity threshold (0-100), default 85

    Returns:
        Tuple containing:
        - Set of indices of similar descriptions
        - List of dictionaries containing similarity details
    """
    from rapidfuzz import fuzz
    import hashlib

    if "job_description" not in df.columns:
        raise ValueError("DataFrame must contain 'job_description' column")

    similar_indices = set()
    similarity_details = []  # Will store detailed comparison information

    # Pre-process descriptions to reduce noise and computation
    print("\nPre-processing descriptions...")
    descriptions = df["job_description"].fillna("").str.lower().tolist()

    # Create initial clusters using hash-based blocking
    print("\nCreating initial clusters...")
    clusters = {}
    for idx, desc in enumerate(descriptions):
        # Create a simhash-like fingerprint using first 100 chars
        fingerprint = hashlib.md5(desc[:FINGERPRINT_LENGTH].encode()).hexdigest()[
            :HASH_PREFIX_LENGTH
        ]
        if fingerprint not in clusters:
            clusters[fingerprint] = []
        clusters[fingerprint].append(idx)

    # Compare descriptions only within same clusters
    print("\nChecking for similar descriptions...")
    for cluster in tqdm(clusters.values(), desc="Processing clusters"):
        if len(cluster) < 2:
            continue

        for i in range(len(cluster)):
            desc1 = descriptions[cluster[i]]
            # Only compare with subsequent descriptions in cluster
            for j in range(i + 1, len(cluster)):
                desc2 = descriptions[cluster[j]]

                # Quick length comparison first
                len_ratio = min(len(desc1), len(desc2)) / max(len(desc1), len(desc2))
                if (
                    len_ratio < LENGTH_RATIO_THRESHOLD
                ):  # Skip if lengths are too different
                    continue

                # Use token_sort_ratio for better handling of word order differences
                similarity = fuzz.token_sort_ratio(desc1, desc2)

                if similarity >= threshold:
                    similar_indices.add(cluster[i])
                    similar_indices.add(cluster[j])

                    # Store detailed information about the similarity match
                    similarity_details.append(
                        {
                            "index1": cluster[i],
                            "index2": cluster[j],
                            "similarity_score": similarity,
                            "job_title1": df.iloc[cluster[i]]["job_title"],
                            "job_title2": df.iloc[cluster[j]]["job_title"],
                            "location1": df.iloc[cluster[i]]["location"],
                            "location2": df.iloc[cluster[j]]["location"],
                            "salary1": df.iloc[cluster[i]]["salary"],
                            "salary2": df.iloc[cluster[j]]["salary"],
                            "description1": df.iloc[cluster[i]]["job_description"],
                            "description2": df.iloc[cluster[j]]["job_description"],
                        }
                    )

    return similar_indices, similarity_details


def is_non_english(text: str, min_score: float = 0.1) -> bool:
    """check if text is likely non-english using fast language detection

    args:
        text: text to analyze
        min_score: minimum confidence score threshold for english (default: 0.1)

    returns:
        true if text is likely non-english, false otherwise
    """
    if not isinstance(text, str) or len(text.strip()) < MIN_TEXT_LENGTH_FOR_LANG_CHECK:
        return False

    try:
        # remove newlines to prevent valueerror
        text = text.replace("\n", " ")
        result = detect(text, low_memory=True)
        # check if english is detected with sufficient confidence
        return not (
            result["lang"] == "en" and result["score"] >= MIN_LANG_CONFIDENCE_SCORE
        )
    except:
        return False


def analyze_glassdoor_data() -> Tuple[pd.DataFrame, Set[int]]:
    # Read the CSV file
    print("\nReading glassdoor.csv...")
    df = pd.read_csv("glassdoor.csv")

    # Ensure string columns are not None
    df["job_description"] = df["job_description"].fillna("")
    df["salary"] = df["salary"].fillna("")
    df["job_title"] = df["job_title"].fillna("")
    df["location"] = df["location"].fillna("")

    # Basic information about the dataset
    print("\n=== BASIC INFORMATION ===")
    print(f"Total number of rows: {len(df)}")
    print(f"Total number of columns: {len(df.columns)}")
    print("\nColumns:", df.columns.tolist())

    # Check for missing values
    print("\n=== MISSING VALUES ===")
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame(
        {
            "Missing Count": missing_values,
            "Missing Percentage": missing_percentages.round(2),
        }
    )
    print(missing_info[missing_info["Missing Count"] > 0])
    print(
        "Not removing rows with missing values in location and job_title. Not necessary for feature extraction"
    )

    # Check for duplicates
    print("\n=== DUPLICATES ===")
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")

    # Check for duplicate job links (same job posted multiple times)
    duplicate_links = df[df.duplicated(subset=["job_link"], keep=False)]
    print(f"Rows with duplicate job links: {len(duplicate_links)}")

    # Value distributions
    print("\n=== VALUE DISTRIBUTIONS ===")
    print("\nCountry distribution:")
    print(df["country"].value_counts())

    print("\nTop 10 job titles:")
    print(df["job_title"].value_counts().head(10))

    # Check for potential data quality issues
    print("\n=== POTENTIAL DATA QUALITY ISSUES ===")

    # Check for non-English content in descriptions
    print("\nChecking for non-English content...")
    tqdm.pandas(desc="Job descriptions")
    desc_mask = df["job_description"].progress_apply(is_non_english)
    non_english_rows = df[desc_mask]
    print(
        f"\nJobs with likely non-English content in descriptions: {len(non_english_rows)}"
    )

    # Check for very short or empty descriptions
    short_desc = df[df["job_description"].str.len() < MIN_DESCRIPTION_LENGTH]
    print(
        f"Jobs with very short descriptions (<{MIN_DESCRIPTION_LENGTH} chars): {len(short_desc)}"
    )

    # Check for invalid salaries (if they don't contain numbers)
    invalid_salaries = df[~df["salary"].str.contains(r"\d", na=True)]
    print(f"Jobs with potentially invalid salaries: {len(invalid_salaries)}")

    # Check for unusual locations
    print("\nUnique locations found:")
    print(df["location"].value_counts().head(10))

    # Find similar job descriptions
    print("\nAnalyzing similar job descriptions...")
    similar_indices, similarity_details = find_similar_descriptions(
        df, threshold=SIMILARITY_THRESHOLD
    )
    print(f"\nFound {len(similar_indices)} jobs with similar descriptions")

    # Create and save detailed similarity report
    if similarity_details:
        similar_df = pd.DataFrame(similarity_details)
        similar_df.sort_values("similarity_score", ascending=False, inplace=True)

        # Save to CSV with detailed comparison information
        output_file = "similar_descriptions_analysis.csv"
        similar_df.to_csv(output_file, index=False)
        print(f"\nSaved detailed similarity analysis to '{output_file}'")

        # Print summary statistics of similarity scores
        print("\nSimilarity Score Statistics:")
        print(similar_df["similarity_score"].describe())

        # Print sample of most similar pairs
        print("\nTop 5 Most Similar Pairs:")
        for _, row in similar_df.head().iterrows():
            print(f"\nSimilarity Score: {row['similarity_score']:.2f}")
            print(f"Title 1: {row['job_title1']} | Title 2: {row['job_title2']}")
            print(f"Location 1: {row['location1']} | Location 2: {row['location2']}")
            print(
                f"Description 1: {row['description1'][:200]} |\n Description 2: {row['description2'][:200]}"
            )

    indices_to_remove = set()
    if similarity_details:
        # Group by index1 to find all similar descriptions
        for details in similarity_details:
            if details["similarity_score"] >= SIMILARITY_THRESHOLD:
                # Always keep index1 and remove index2
                indices_to_remove.add(details["index2"])

    # Remove the similar descriptions from the dataframe
    df.drop(index=indices_to_remove, inplace=True)
    print(f"\nRemoved {len(indices_to_remove)} similar descriptions")

    # Save problematic entries to a separate CSV for review
    # Convert columns to pandas Series explicitly to ensure proper method access
    job_desc_series = pd.Series(df["job_description"], dtype=str)
    salary_series = pd.Series(df["salary"], dtype=str)

    problematic = df[
        (df.isnull().any(axis=1))  # Any missing values
        | (df.duplicated())  # Duplicates
        | (
            job_desc_series.fillna("").str.len() < MIN_DESCRIPTION_LENGTH
        )  # Short descriptions
        | (~salary_series.fillna("").str.contains(r"\d"))  # Invalid salaries
        | (job_desc_series.apply(is_non_english))  # Non-English content in description
    ].copy()  # Create a copy to avoid SettingWithCopyWarning

    if len(problematic) > 0:
        # Add a column to indicate why entries are problematic
        problematic.loc[:, "issues"] = ""
        problematic.loc[problematic.isnull().any(axis=1), "issues"] += "missing_values;"
        problematic.loc[problematic.duplicated(), "issues"] += "duplicate_entry;"
        problematic.loc[
            problematic["job_description"].fillna("").str.len()
            < MIN_DESCRIPTION_LENGTH,
            "issues",  # type: ignore
        ] += "short_description;"
        problematic.loc[
            ~problematic["salary"].fillna("").str.contains(r"\d"), "issues"  # type: ignore
        ] += "invalid_salary;"
        problematic.loc[problematic["job_title"] == INVALID_TITLE, "issues"] += (
            "invalid_title;"
        )
        print("\nTagging problematic entries...")

        print("Checking job descriptions for non-English content...")
        tqdm.pandas(desc="Checking job descriptions")
        problematic.loc[
            problematic["job_description"].progress_apply(is_non_english), "issues"  # type: ignore
        ] += "non_english_description;"

        problematic.to_csv("problematic_entries.csv", index=True)
        print(
            f"\nSaved {len(problematic)} problematic entries to 'problematic_entries.csv'"
        )

        # Print summary of issues
        print("\nBreakdown of issues:")
        for issue in [
            "missing_values",
            "duplicate_entry",
            "short_description",
            "invalid_salary",
            "non_english_description",
            "invalid_title",
        ]:
            count = problematic["issues"].str.contains(issue).sum()  # type: ignore
            print(f"- {issue}: {count} entries")

    # Remove problematic entries
    print("\nRemoving problematic entries...")
    clean_df = df[
        ~(df.duplicated())  # Remove duplicates
        & (
            df["job_description"].str.len() >= MIN_DESCRIPTION_LENGTH
        )  # Remove short descriptions
        & (df["salary"].str.contains(r"\d", na=False))  # Remove invalid salaries
        & (df["job_title"] != INVALID_TITLE)  # Remove invalid titles
        & ~(df.index.isin(indices_to_remove))  # Remove similar descriptions
        & ~(desc_mask)  # Remove non-English content
    ]

    print(f"\nRemoved {len(df) - len(clean_df)} total problematic entries")

    # Save cleaned data to CSV
    print("\nSaving cleaned data...")
    clean_df.to_csv("cleaned_glassdoor.csv", index=False)
    print(f"Saved {len(clean_df)} cleaned records to 'cleaned_glassdoor.csv'")

    return clean_df, indices_to_remove


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")  # Claude's encoding
    return len(encoding.encode(text))


def calculate_claude_cost(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    """Calculate Claude API cost based on token usage"""
    # https://www.anthropic.com/pricing#anthropic-api
    input_cost_per_1m = CLAUDE_INPUT_COST_PER_1M
    output_cost_per_1m = CLAUDE_OUTPUT_COST_PER_1M

    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(total_cost, 4),
    }


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Create stopwords set once
STOP_WORDS = set(stopwords.words("english"))
# Compile regex pattern once
WORD_PATTERN = re.compile(r"\b\w+\b")


def remove_stopwords(text: str) -> str:
    """Remove stopwords from text using regex splitting"""
    if not isinstance(text, str):
        return ""
    # Use regex to split text into words and filter stopwords
    return " ".join(
        word for word in WORD_PATTERN.findall(text.lower()) if word not in STOP_WORDS
    )


def analyze_token_usage():
    """Analyze token usage and cost for EDA processing"""
    print("\n=== TOKEN USAGE ANALYSIS ===")

    # Read the CSV file
    df = pd.read_csv("glassdoor.csv")

    # Initialize counters
    original_input_tokens = 0
    original_output_tokens = 0
    filtered_input_tokens = 0
    filtered_output_tokens = 0

    print("\nCalculating original token usage...")

    template_tokens = count_tokens(TEMPLATE)
    print(f"\nTemplate tokens (per request): {template_tokens}")

    # First calculate original tokens with progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Counting original tokens"):
        original_prompt = f"""
        Job Title: {row['job_title']}
        Location: {row['location']}
        Salary: {row['salary']}
        Description: {row['job_description']}
        """

        # Calculate tokens for original text
        original_input = template_tokens + count_tokens(original_prompt)
        original_input_tokens += original_input

    # Calculate and display original costs first
    original_cost = calculate_claude_cost(original_input_tokens, original_output_tokens)

    print("\n=== ORIGINAL TEXT TOKEN USAGE AND COST ===")
    print(f"Input Tokens: {original_cost['input_tokens']:,}")
    print(f"Output Tokens: {original_cost['output_tokens']:,}")
    print(f"Input Cost: ${original_cost['input_cost']:,.2f}")
    print(f"Output Cost: ${original_cost['output_cost']:,.2f}")
    print(f"Total Cost: ${original_cost['total_cost']:,.2f}")

    print("\nCalculating filtered token usage (removing stopwords)...")

    # Now calculate filtered tokens with progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Removing stopwords"):
        filtered_prompt = f"""
        Job Title: {remove_stopwords(str(row['job_title']))}
        Location: {remove_stopwords(str(row['location']))}
        Salary: {str(row['salary'])}
        Description: {remove_stopwords(str(row['job_description']))}
        """

        # Calculate tokens for filtered text
        filtered_input = template_tokens + count_tokens(filtered_prompt)
        filtered_input_tokens += filtered_input

        # Estimate output tokens (same for both since output format doesn't change)
        sample_output = {
            "soft_skills": ["communication", "teamwork"],
            "hard_skills": ["python", "sql"],
            "location_flexibility": "remote",
            "contract_type": "full-time",
            "education_level": "bachelors",
            "field_of_study": "computer science",
            "seniority": "mid-level",
            "min_years_experience": 3,
            "min_salary": 80000,
            "max_salary": 120000,
            "salary_currency": "USD",
            "salary_period": "yearly",
        }
        output_tokens = count_tokens(json.dumps(sample_output))
        original_output_tokens += output_tokens
        filtered_output_tokens += output_tokens

    # Calculate costs for original text
    original_cost = calculate_claude_cost(original_input_tokens, original_output_tokens)

    # Calculate costs for filtered text
    filtered_cost = calculate_claude_cost(filtered_input_tokens, filtered_output_tokens)

    print("\n=== FILTERED TEXT (STOPWORDS REMOVED) TOKEN USAGE AND COST ===")
    print(f"Input Tokens: {filtered_cost['input_tokens']:,}")
    print(f"Output Tokens: {filtered_cost['output_tokens']:,}")
    print(f"Input Cost: ${filtered_cost['input_cost']:,.2f}")
    print(f"Output Cost: ${filtered_cost['output_cost']:,.2f}")
    print(f"Total Cost: ${filtered_cost['total_cost']:,.2f}")

    # Calculate and display savings
    token_reduction = original_cost["input_tokens"] - filtered_cost["input_tokens"]
    cost_savings = original_cost["total_cost"] - filtered_cost["total_cost"]
    reduction_percentage = (token_reduction / original_cost["input_tokens"]) * 100

    print("\n=== SAVINGS ANALYSIS ===")
    print(f"Token Reduction: {token_reduction:,} tokens ({reduction_percentage:.1f}%)")
    print(f"Cost Savings: ${cost_savings:.2f}")

    # The cost is not substantial, around 22.2% savings. I would prefer to retain the stopwords so that the GenAI understands job desription.


def recalculate_analysis(df: pd.DataFrame) -> None:
    """Recalculate basic information and distributions after cleaning"""
    print("\n=== RECALCULATED ANALYSIS AFTER CLEANING ===")

    # Basic information
    print("\n=== UPDATED BASIC INFORMATION ===")
    print(f"Total number of rows: {len(df)}")
    print(f"Total number of columns: {len(df.columns)}")

    # Updated value distributions
    print("\n=== UPDATED VALUE DISTRIBUTIONS ===")
    print("\nCountry distribution:")
    print(df["country"].value_counts())

    print("\nTop 10 job titles:")
    print(df["job_title"].value_counts().head(10))

    print("\n=== RAW DATA TOKEN USAGE ===")
    analyze_token_usage()

    # Recalculate token usage and costs
    print("\n=== RECALCULATED TOKEN USAGE ===")
    template_tokens = count_tokens(TEMPLATE)  # Same template as before

    input_tokens = 0
    output_tokens = 0

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Calculating cleaned token usage"
    ):
        cleaned_prompt = f"""
        Job Title: {str(row['job_title'])}
        Location: {str(row['location'])}
        Salary: {str(row['salary'])}
        Description: {str(row['job_description'])}
        """

        input_tokens += template_tokens + count_tokens(cleaned_prompt)

        # Estimate output tokens using sample output
        sample_output = {
            "soft_skills": ["communication", "teamwork"],
            "hard_skills": ["python", "sql"],
            "location_flexibility": "remote",
            "contract_type": "full-time",
            "education_level": "bachelors",
            "field_of_study": "computer science",
            "seniority": "mid-level",
            "min_years_experience": 3,
            "min_salary": 80000,
            "max_salary": 120000,
            "salary_currency": "USD",
            "salary_period": "yearly",
        }
        output_tokens += count_tokens(json.dumps(sample_output))

    # Calculate final costs
    final_cost = calculate_claude_cost(input_tokens, output_tokens)

    print("\n=== FINAL TOKEN USAGE AND COST ===")
    print(f"Input Tokens: {final_cost['input_tokens']:,}")
    print(f"Output Tokens: {final_cost['output_tokens']:,}")
    print(f"Input Cost: ${final_cost['input_cost']:,.2f}")
    print(f"Output Cost: ${final_cost['output_cost']:,.2f}")
    print(f"Total Cost: ${final_cost['total_cost']:,.2f}")


if __name__ == "__main__":
    # Run initial analysis which includes finding and removing similar descriptions
    clean_df, indices_to_remove = analyze_glassdoor_data()

    # Recalculate analysis with clean data
    recalculate_analysis(clean_df)
