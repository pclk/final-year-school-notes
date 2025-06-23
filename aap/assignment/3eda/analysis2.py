import pandas as pd
from ydata_profiling import ProfileReport
from tqdm import tqdm

# Constants
INPUT_FILE = "../2feature_extraction/jobs.csv"
OUTPUT_FILE = "cleaned_jobs.csv"
PROFILE_OUTPUT = "jobs_profile_report.html"

FREQUENCY_THRESHOLD = 1  # 0-100 probability slider for removal
SGD_TO_USD = 0.74
# More precise representation of the conversion rate
RUP_TO_USD = 12 / 1000  # This ensures we keep precision
MIN_RATE_PER_HOUR_RUP = 100

# Read the data
df = pd.read_csv(INPUT_FILE)


# Remove unnecessary columns
columns_to_drop = [
    "job_id",  # Internal identifier not needed
    "job_link",  # URL not needed for analysis
    "location",  # Location not needed
    "index",  # If exists
    "Unnamed: 0",  # If exists
]

df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])


# Clean and process the data
def clean_skills_list(skills_str):
    if pd.isna(skills_str) or skills_str == "":
        return []
    # Replace both spaces and underscores with hyphens within each skill
    return [
        skill.strip().lower().replace(" ", "-").replace("_", "-")
        for skill in skills_str.split(",")
    ]


# Process skills columns
df["soft_skills"] = df["soft_skills"].apply(clean_skills_list)
df["hard_skills"] = df["hard_skills"].apply(clean_skills_list)
df["field_of_study"] = df["field_of_study"].apply(clean_skills_list)

# Replace -1 values with NaN for experience
df["min_years_experience"] = df["min_years_experience"].replace(-1, pd.NA)

# Impute empty experience values with 0.0
print(
    f"\nEmpty experience values before imputation: {df['min_years_experience'].isna().sum()}"
)
df["min_years_experience"] = df["min_years_experience"].fillna(0.0)
print(
    f"Empty experience values after imputation: {df['min_years_experience'].isna().sum()}"
)

# Fill unspecified values
df["location_flexibility"] = df["location_flexibility"].replace("unspecified", pd.NA)
df["contract_type"] = df["contract_type"].replace("unspecified", pd.NA)
df["education_level"] = df["education_level"].replace("unspecified", pd.NA)
df["seniority"] = df["seniority"].replace("unspecified", pd.NA)

# Analyze and clean low-frequency values
categorical_cols = [
    "location_flexibility",
    "contract_type",
    "education_level",
    "seniority",
    "salary_period",
    # "soft_skills",
    # "hard_skills",
    # "field_of_study",
]

print("\nLeast frequent values in categorical columns:")
for col in categorical_cols:
    value_counts = df[col].value_counts(normalize=True) * 100
    print(f"\n{col.replace('_', ' ').title()}:")
    print(value_counts.nsmallest(6).round(2).to_string())

print(
    f"\nReplacing values that occur less than {FREQUENCY_THRESHOLD}% of the time with NA..."
)

# Store original counts
original_counts = {col: df[col].notna().sum() for col in categorical_cols}

# Replace low-frequency values with NA
for col in categorical_cols:
    value_counts = df[col].value_counts(normalize=True) * 100
    low_freq_values = value_counts[value_counts < FREQUENCY_THRESHOLD].index
    df[col] = df[col].replace(dict.fromkeys(low_freq_values, pd.NA))

# Print removal report
print("\nRemoval Report:")
for col in categorical_cols:
    removed = original_counts[col] - df[col].notna().sum()
    if removed > 0:
        print(f"{col.replace('_', ' ').title()}: {removed} values replaced with NA")

# Print some basic statistics
print("\nDataset Summary:")
print(f"Total number of jobs: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nMissing values:")
print(df.isna().sum())

# Analyze salary_period NA values
print("\nSalary Period Analysis:")
print(f"Total NA values in salary_period: {df['salary_period'].isna().sum()}")
print("\nSample of rows with NA salary_period:")
print(df[df["salary_period"].isna()].head())

# Analyze specific cases
print("\nAnalyzing specific salary cases with missing periods:")
print("\nREMOVAL NOTICE:")
print("The Astronomer (Adjunct Faculty) position will be removed because:")
print("- The salary ($1,700 - $3,000) is likely per-course based")
print("- Per-course compensation cannot be standardized with other salary periods")
print("- This ensures consistency in salary comparisons across the dataset")

# Remove the specific astronomer row by index
df = df[df.index != 9737]
print(
    "\nRemoved row with index 9737 (Astronomer position with per-course salary structure)"
)

print("\nSalary Period Analysis for University Lecturer case:")
print("Salary Range: $7,500 - $8,000")
print("Likely Period: MONTHLY")
print("Reasoning:")
print("- Part-time lecturer positions typically quote monthly salaries")
print("- The range is too low for yearly academic salary")
print("- Too high for hourly or weekly compensation")
print("- Consistent with typical monthly lecturer compensation")

# Manually set the salary period for the University Lecturer case
df.loc[27675, "salary_period"] = "monthly"
print("\nUpdated salary period for University Lecturer (index 27675) to monthly")

# Convert salaries to USD based on country
print("\nConverting salaries to USD...")

# Create USD columns
df["min_salary_usd"] = df["min_salary"]
df["max_salary_usd"] = df["max_salary"]

# Convert Singapore salaries (SGD to USD)
sg_mask = df["country"] == "SG"
df.loc[sg_mask, "min_salary_usd"] = df.loc[sg_mask, "min_salary"] * SGD_TO_USD
df.loc[sg_mask, "max_salary_usd"] = df.loc[sg_mask, "max_salary"] * SGD_TO_USD


# Function to parse Indian salary format
def parse_indian_salary(salary_text):
    if pd.isna(salary_text):
        return None, None, None

    import re

    # Pattern to match Indian salary format with commas and decimals
    # Handles both regular amounts and hourly rates with decimals
    pattern = r"₹\s*([\d,]+(?:\.\d{1,2})?)\s*([LKT])?(?:\s*-\s*₹\s*([\d,]+(?:\.\d{1,2})?)\s*([LKT])?)?(?:\s*Per hour)?"

    # Print reasoning for very low hourly rates
    if "Per hour" in salary_text or "per hour" in salary_text:
        rate = float(re.findall(r"₹\s*([\d.]+)", salary_text)[0])
        if rate < MIN_RATE_PER_HOUR_RUP:  # Less than ₹100 per hour
            print(f"\nFiltering out suspicious hourly rate: {salary_text}")
            print("Reasoning:")
            print(f"1. Rate of ₹{rate:.2f} per hour is unrealistically low")
            print(
                f"2. This converts to approximately ${(rate * RUP_TO_USD):.2f} USD per hour"
            )
            print("3. This is below minimum wage standards for technical roles")
            print("4. Likely a data entry error or misclassified compensation period")
            return None, None, None

    match = re.search(pattern, salary_text)
    if not match:
        return None, None, None

    min_val, min_unit, max_val, max_unit = match.groups()

    def convert_to_rupees(value, unit):
        if value is None:
            return None
        # Remove commas and convert to float
        value = float(value.replace(",", ""))
        if unit == "L":  # Lakh = 100,000
            return value * 100000
        elif unit in ["K", "T"]:  # Both K and T represent Thousand = 1,000
            return value * 1000
        return value  # If no unit, assume it's direct value

    min_salary = convert_to_rupees(min_val, min_unit)
    max_salary = convert_to_rupees(max_val, max_unit) if max_val else min_salary

    # Detect if it's per hour - check both "Per hour" and "per hour"
    is_hourly = any(phrase in salary_text for phrase in ["Per hour", "per hour"])

    return min_salary, max_salary, "hourly" if is_hourly else "yearly"


# Process Indian salaries
print("\nProcessing Indian Rupee salaries...")
in_mask = df["country"] == "IN"
indian_salaries = df[in_mask].copy()

# Store original values for verification
original_values = indian_salaries[
    ["salary", "min_salary", "max_salary", "salary_period"]
].copy()

# Parse each Indian salary
parsed_salaries = indian_salaries["salary"].apply(parse_indian_salary)
indian_salaries["min_salary"] = parsed_salaries.apply(lambda x: x[0])
indian_salaries["max_salary"] = parsed_salaries.apply(lambda x: x[1])
indian_salaries["salary_period"] = parsed_salaries.apply(lambda x: x[2])

# Convert to USD with better precision
indian_salaries["min_salary_usd"] = indian_salaries["min_salary"].apply(
    lambda x: (x * 12) / 1000 if pd.notna(x) else x
)
indian_salaries["max_salary_usd"] = indian_salaries["max_salary"].apply(
    lambda x: (x * 12) / 1000 if pd.notna(x) else x
)

# Update the main dataframe with explicit assignment
df.loc[in_mask, "min_salary_usd"] = indian_salaries["min_salary_usd"]
df.loc[in_mask, "max_salary_usd"] = indian_salaries["max_salary_usd"]
df.loc[in_mask, "salary_period"] = indian_salaries["salary_period"]

# Verify the update worked
print("\nVerifying updates:")
verify_idx = 178  # One of our problem indices
print(f"Values for index {verify_idx}:")
print(f"In indian_salaries DataFrame:")
print(f"min_salary_usd: {indian_salaries.loc[verify_idx, 'min_salary_usd']}")
print(f"In main DataFrame:")
print(f"min_salary_usd: {df.loc[verify_idx, 'min_salary_usd']}")

# Debug specific indexes
debug_indexes = [178, 401, 767]
print("\nDebugging specific salary conversions:")
for idx in debug_indexes:
    row = df.loc[idx]
    print(f"\nIndex {idx}:")
    print(f"Country: {row['country']}")
    print(f"Original salary text: {row['salary']}")
    print(f"min_salary (original): {row['min_salary']}")
    print(f"max_salary (original): {row['max_salary']}")
    print(f"min_salary_usd: {row['min_salary_usd']}")
    print(f"max_salary_usd: {row['max_salary_usd']}")
    if row["country"] == "IN":
        print(f"Conversion calculation:")
        print(
            f"min: ({row['min_salary']} * 12) / 1000 = {(row['min_salary'] * 12) / 1000}"
        )
        print(
            f"max: ({row['max_salary']} * 12) / 1000 = {(row['max_salary'] * 12) / 1000}"
        )
    elif row["country"] == "SG":
        print(f"Conversion calculation:")
        print(
            f"min: {row['min_salary']} * {SGD_TO_USD} = {row['min_salary'] * SGD_TO_USD}"
        )
        print(
            f"max: {row['max_salary']} * {SGD_TO_USD} = {row['max_salary'] * SGD_TO_USD}"
        )

# Print summary of changes
print(f"\nProcessed {in_mask.sum()} Indian salary entries")
print("\nSample of parsed Indian salaries:")
sample_size = min(5, len(indian_salaries))
for _, row in indian_salaries.sample(sample_size).iterrows():
    print(f"\nOriginal: {row['salary']}")
    print(
        f"Parsed: ₹{row['min_salary']:,.2f} - ₹{row['max_salary']:,.2f} ({row['salary_period']})"
    )
    print(f"In USD: ${row['min_salary_usd']:,.2f} - ${row['max_salary_usd']:,.2f}")

print(f"Converted {sg_mask.sum()} Singapore salaries from SGD to USD")
print(f"Converted {in_mask.sum()} India salaries from Rupee to USD")

# Define salary thresholds for different periods
SALARY_THRESHOLDS = {
    "hourly": {
        "min": 0,  # No minimum for hourly rate
        "max": 500,  # $500/hour maximum
    },
    "monthly": {
        "min": 500,  # $500/month minimum
        "max": 50_000,  # $50000/month maximum
    },
    "yearly": {
        "min": 6000,  # $6000/year minimum
        "max": 3_000_000,  # $3 mill/year maximum
    },
}


def validate_and_correct_salary_period(row):
    """
    Validates and corrects salary period based on salary ranges in USD.
    Returns the most likely salary period.
    """
    min_salary = row["min_salary_usd"]
    max_salary = row["max_salary_usd"]
    current_period = row["salary_period"]

    # Use the higher salary for validation to catch edge cases
    test_salary = max(min_salary, max_salary) if max_salary > 0 else min_salary

    # Skip if salary is invalid
    if test_salary <= 0:
        return current_period

    # Check if current period is valid
    if pd.notna(current_period):
        thresh = SALARY_THRESHOLDS.get(current_period)
        if thresh and thresh["min"] <= test_salary <= thresh["max"]:
            return current_period

    # Try to determine the correct period
    for period, thresh in SALARY_THRESHOLDS.items():
        if thresh["min"] <= test_salary <= thresh["max"]:
            return period

    # If no period matches, use yearly as default for outliers
    return "yearly"


# Validate and correct salary periods
print("\nValidating and correcting salary periods...")
original_periods = df["salary_period"].copy()
df["salary_period"] = df.apply(validate_and_correct_salary_period, axis=1)

# Print summary of changes
period_changes = (original_periods != df["salary_period"]).sum()
print(f"\nSalary period corrections made: {period_changes}")
print("\nPeriod distribution before correction:")
print(original_periods.value_counts(dropna=False))
print("\nPeriod distribution after correction:")
print(df["salary_period"].value_counts(dropna=False))

# Define conversion multipliers
HOURS_PER_YEAR = 2080  # 40 hours/week * 52 weeks
MONTHS_PER_YEAR = 12

# Create columns for standardized salaries in USD
df["yearly_min_salary"] = df["min_salary_usd"]
df["yearly_max_salary"] = df["max_salary_usd"]

# Convert hourly salaries to yearly
hourly_mask = df["salary_period"] == "hourly"
df.loc[hourly_mask, "yearly_min_salary"] = (
    df.loc[hourly_mask, "min_salary_usd"] * HOURS_PER_YEAR
)
df.loc[hourly_mask, "yearly_max_salary"] = (
    df.loc[hourly_mask, "max_salary_usd"] * HOURS_PER_YEAR
)

# Convert monthly salaries to yearly
monthly_mask = df["salary_period"] == "monthly"
df.loc[monthly_mask, "yearly_min_salary"] = (
    df.loc[monthly_mask, "min_salary_usd"] * MONTHS_PER_YEAR
)
df.loc[monthly_mask, "yearly_max_salary"] = (
    df.loc[monthly_mask, "max_salary_usd"] * MONTHS_PER_YEAR
)

# Calculate salary midpoints
print("\nCalculating salary midpoints...")

# Initialize midpoint column
df["yearly_salary_midpoint"] = pd.NA

# Calculate midpoints for rows where both min and max are valid
valid_range_mask = (df["yearly_min_salary"] >= 0) & (df["yearly_max_salary"] >= 0)
df.loc[valid_range_mask, "yearly_salary_midpoint"] = df.loc[
    valid_range_mask, ["yearly_min_salary", "yearly_max_salary"]
].mean(axis=1)

# For rows where only one value is valid, use that value
min_only_mask = (df["yearly_min_salary"] >= 0) & (df["yearly_max_salary"] < 0)
df.loc[min_only_mask, "yearly_salary_midpoint"] = df.loc[
    min_only_mask, "yearly_min_salary"
]

max_only_mask = (df["yearly_max_salary"] >= 0) & (df["yearly_min_salary"] < 0)
df.loc[max_only_mask, "yearly_salary_midpoint"] = df.loc[
    max_only_mask, "yearly_max_salary"
]

# Set manual thresholds
print("\nApplying manual salary thresholds...")
SALARY_LOW_THRESHOLD = 1000  # $1000 per year
SALARY_LOW_THRESHOLD_DISPLAY = "1K"
SALARY_HIGH_THRESHOLD = 1000000  # $2 million per year
SALARY_HIGH_THRESHOLD_DISPLAY = "1M"

# Identify extreme cases based on manual thresholds
extreme_salaries = df[
    (df["yearly_salary_midpoint"] >= SALARY_HIGH_THRESHOLD)
    | (df["yearly_salary_midpoint"] <= SALARY_LOW_THRESHOLD)
]
# Add a column to identify if it's high or low
extreme_salaries["salary_category"] = "NORMAL"
extreme_salaries.loc[
    extreme_salaries["yearly_salary_midpoint"] >= SALARY_HIGH_THRESHOLD,
    "salary_category",
] = f"HIGH (Above ${SALARY_HIGH_THRESHOLD_DISPLAY}/year)"
extreme_salaries.loc[
    extreme_salaries["yearly_salary_midpoint"] <= SALARY_LOW_THRESHOLD,
    "salary_category",
] = f"LOW (Below ${SALARY_LOW_THRESHOLD_DISPLAY}/year)"

extreme_salaries.to_csv("extreme_salaries.csv", index=True)
print(f"Exported {len(extreme_salaries)} extreme salary cases to extreme_salaries.csv")
print("\nYearly Salary Thresholds (manual):")
print(
    f"Salary thresholds: ${SALARY_LOW_THRESHOLD:,.2f} (minimum) to ${SALARY_HIGH_THRESHOLD:,.2f} (maximum)"
)
print(f"Total extreme salaries found: {len(extreme_salaries)}")
print(
    f"High salaries (>${SALARY_HIGH_THRESHOLD_DISPLAY}): {(extreme_salaries['salary_category'] == f'HIGH (Above ${SALARY_HIGH_THRESHOLD_DISPLAY}/year)').sum()}"
)
print(
    f"Low salaries (<${SALARY_LOW_THRESHOLD_DISPLAY}): {(extreme_salaries['salary_category'] == f'LOW (Below ${SALARY_LOW_THRESHOLD_DISPLAY}/year)').sum()}"
)

# Print some statistics about the midpoint calculations
print("\nMidpoint Calculation Statistics:")
print(
    f"Total rows with calculated midpoints: {df['yearly_salary_midpoint'].notna().sum()}"
)
print(f"Rows using both min and max: {valid_range_mask.sum()}")
print(f"Rows using only min salary: {min_only_mask.sum()}")
print(f"Rows using only max salary: {max_only_mask.sum()}")

# Print conversion summary
print("\nSalary Conversion Summary:")
print(f"Hourly salaries converted: {hourly_mask.sum()}")
print(f"Monthly salaries converted: {monthly_mask.sum()}")
print(f"Already yearly salaries: {(df['salary_period'] == 'yearly').sum()}")
print(f"NA or other periods: {df['salary_period'].isna().sum()}")

# Print specific case conversion result
lecturer_row = df.loc[27675]
print("\nUniversity Lecturer conversion result:")
print(
    f"Original salary range: ${lecturer_row['min_salary_usd']:,.2f} - ${lecturer_row['max_salary_usd']:,.2f} (monthly)"
)
print(
    f"Converted yearly range: ${lecturer_row['yearly_min_salary']:,.2f} - ${lecturer_row['yearly_max_salary']:,.2f}"
)


# Update salary for index 9763. misread as 600k max salary and 3 million max salary
df.loc[9763, "min_salary"] = 113_000
df.loc[9763, "max_salary"] = 183_000

# Set experience thresholds
print("\nAnalyzing extreme experience requirements...")
EXPERIENCE_HIGH_THRESHOLD = 25  # > 25 years

# Interestingly there are experience in float values. This are found when the recruiter has listed the following examples:
# - row index 28725: ...0.1 to 2 years of business operations experience ... (0.1 min_years_experience)
# - row index 28482: ...Minimum 2 month of experience in financial services ... (0.17 min_years_experience)
# - row index 5674: ...EXPERIENCE -MINIMUM 3 MONTHS ... (0.25 min_years_experience)
# The feature extraction model (claude 3.5 haiku) is able to calculate float values for experience by dividing the
# months by 12.
#
# The model is able to do addition:
# - row index 35927: ... 12 or more years of experience in credentialed journalism
# with additional 5 years or more experience in corporate communications... (17 min_years_experience)
# By looking at the extreme_experience.csv, we can see that recruiters really list 25 years of experience in their job listings.


# However, the rows where the experience has exceeded threshold is mistakenly calculated by the model.
# - row index 12575: ...Must have accrued a minimum of 2,000 hours as a Pilot in Command... (2000 min_years_experience)

# Manually calculate years of experience in row 12575
print("\nCalculating correct experience for Pilot position (row 12575):")
FLIGHT_HOURS = 2000  # Required flight hours
AVG_FLIGHT_HOURS_PER_YEAR = 800  # Average commercial pilot flies ~800 hours per year
years_experience = FLIGHT_HOURS / AVG_FLIGHT_HOURS_PER_YEAR
print(f"Flight hours required: {FLIGHT_HOURS}")
print(f"Average flight hours per year: {AVG_FLIGHT_HOURS_PER_YEAR}")
print(f"Calculated years of experience: {years_experience:.1f} years")

# Update the incorrect value in the dataframe
df.loc[12575, "min_years_experience"] = years_experience
print(
    f"Updated row 12575 min_years_experience from 2000 to {years_experience:.1f} years"
)

# Identify extreme cases based on experience thresholds
extreme_experience = df[
    (df["min_years_experience"] >= EXPERIENCE_HIGH_THRESHOLD)
    | ((df["min_years_experience"] > 0) & (df["min_years_experience"] < 1))
]

# Add a column to identify if it's high or low experience requirement
extreme_experience["experience_category"] = "NORMAL"
extreme_experience.loc[
    extreme_experience["min_years_experience"] >= EXPERIENCE_HIGH_THRESHOLD,
    "experience_category",
] = f"HIGH (Above {EXPERIENCE_HIGH_THRESHOLD}+ years)"
extreme_experience.loc[
    (extreme_experience["min_years_experience"] > 0)
    & (extreme_experience["min_years_experience"] < 1),
    "experience_category",
] = f"ENTRY (Less than 1 year)"

extreme_experience.to_csv("extreme_experience.csv", index=True)
print(
    f"Exported {len(extreme_experience)} extreme experience cases to extreme_experience.csv"
)
print("\nYears of Experience Thresholds:")
print(
    f"Experience thresholds: floating values between 0 to 1 years and those above {EXPERIENCE_HIGH_THRESHOLD}+ years "
)
print(f"Total extreme experience requirements found: {len(extreme_experience)}")
print(
    f"High experience ({EXPERIENCE_HIGH_THRESHOLD}+ years): {(extreme_experience['experience_category'] == f'HIGH (Above {EXPERIENCE_HIGH_THRESHOLD}+ years)').sum()}"
)
print(
    f"Entry level (No experience): {(extreme_experience['experience_category'] == f'ENTRY (No experience required)').sum()}"
)


# print("\nPreparing data for profile report...")
# df_profile = df.copy()
#
# # Convert skills lists to space-separated strings
# # Skills are already underscore-separated from the cleaning step
# df_profile["soft_skills"] = df_profile["soft_skills"].apply(
#     lambda x: " ".join(x) if isinstance(x, list) else ""
# )
# df_profile["hard_skills"] = df_profile["hard_skills"].apply(
#     lambda x: " ".join(x) if isinstance(x, list) else ""
# )
# df_profile["field_of_study"] = df_profile["field_of_study"].apply(
#     lambda x: " ".join(x) if isinstance(x, list) else ""
# )
#
# # Generate profile report
# print("Generating profile report...")
# profile = ProfileReport(df_profile, title="Jobs Dataset Profiling Report")
# profile.to_file(PROFILE_OUTPUT)
# print(f"Profile report saved to {PROFILE_OUTPUT}")
#
# Remove columns that are no longer needed after processing
print("\nRemoving unnecessary columns:")
columns_to_remove = [
    "query",  # Search term used for scraping and feature extraction - not relevant for modelling
    "job_title",  # used for feature extraction - not revelant for modelling
    "job_description",  # Raw text data - already processed into skills
    "salary",  # Original salary text - already parsed into standardized values
    "salary_period",  # Original period - already converted to yearly values
    # Salaries - already converted to yearly_salary_midpoint
    "max_salary",
    "min_salary",
    "min_salary_usd",
    "max_salary_usd",
    "yearly_min_salary",
    "yearly_max_salary",
]

print("Removing columns with reasoning:")
for col in columns_to_remove:
    if col in df.columns:
        print(f"- {col}: Already processed/normalized into other columns")

df = df.drop(columns=columns_to_remove)

# Perform skill similarity analysis using RapidFuzz
print("\nPerforming skill similarity analysis...")


# Cache for normalized skills
_normalized_cache = {}


def analyze_skill_similarities(skills_list, output_file, similarity_threshold=0):
    """
    Simple analysis of skill similarities using RapidFuzz's built-in ratio scorer
    Args:
        skills_list: List of lists containing skills
        output_file: Where to save the results
        similarity_threshold: Minimum similarity score (0-100) to consider
    Returns:
        DataFrame with similar skills and their scores
    """
    from rapidfuzz import process, fuzz
    import numpy as np
    from tqdm import tqdm
    from collections import defaultdict

    # Get unique skills using set comprehension
    unique_skills = sorted(set(skill for skills in skills_list for skill in skills))
    if not unique_skills:
        print("No skills found to analyze")
        return None

    print(f"Analyzing {len(unique_skills)} unique skills...")

    # Use dictionaries for faster updates
    similar_skills_dict = defaultdict(list)
    similar_scores_dict = defaultdict(list)

    # Process all skills
    for skill in tqdm(unique_skills, desc="Processing skills"):
        # Get similarities using fuzz.ratio
        matches = process.extract(
            skill,
            unique_skills,
            scorer=fuzz.ratio,
            limit=11,
            score_cutoff=similarity_threshold,
        )

        # Process matches efficiently
        skills, scores = [], []
        for match, score, _ in matches:
            if match != skill:
                skills.append(match)
                scores.append(score)

        # Pad lists if needed
        skills.extend([np.nan] * (10 - len(skills)))
        scores.extend([np.nan] * (10 - len(scores)))

        # Store results
        similar_skills_dict[skill] = skills[:10]
        similar_scores_dict[skill] = scores[:10]

    # Create final DataFrame efficiently
    result_data = {}
    for i in range(10):
        result_data[f"similar_{i+1}"] = [
            similar_skills_dict[skill][i] for skill in unique_skills
        ]
        result_data[f"score_{i+1}"] = [
            similar_scores_dict[skill][i] for skill in unique_skills
        ]

    result_df = pd.DataFrame(result_data, index=unique_skills)
    result_df.index.name = "skill"

    # Save results
    result_df.to_csv(output_file)
    print(f"Saved similarity analysis to {output_file}")

    # Print sample results with sorted similarities
    print("\nExample similar skills:")
    for skill in unique_skills[:3]:
        print(f"\nSimilar to '{skill}':")
        # Create pairs of (similar_skill, score) and filter out NaN values
        pairs = [
            (similar_skills_dict[skill][i], similar_scores_dict[skill][i])
            for i in range(len(similar_skills_dict[skill]))
            if pd.notna(similar_skills_dict[skill][i])
            and pd.notna(similar_scores_dict[skill][i])
        ]
        # Sort by score in descending order
        pairs.sort(key=lambda x: x[1], reverse=True)
        # Print top 3 most similar
        for similar, score in pairs[:3]:
            print(f"  - {similar} (similarity: {score:.1f}%)")

    return result_df


# Analyze each skill type separately
print("\nAnalyzing soft skills...")
soft_skills_df = analyze_skill_similarities(
    df["soft_skills"], "soft_skills_similarity.csv"
)

print("\nAnalyzing hard skills...")
hard_skills_df = analyze_skill_similarities(
    df["hard_skills"], "hard_skills_similarity.csv"
)

print("\nAnalyzing fields of study...")
field_skills_df = analyze_skill_similarities(
    df["field_of_study"], "field_similarity.csv"
)

# soft_skills_similarity
# 97.5 % similarity and above, change to shorter phrase.
#
#
# consider using a model to determine semantic similarity
# cooperation vs coordination


# Function to create replacement mapping based on similarity analysis
def create_skill_replacements(similarity_df, similarity_threshold=97.5):
    """
    Create a mapping of skills to be replaced based on high similarity scores
    Args:
        similarity_df: DataFrame containing similarity analysis
        similarity_threshold: Minimum similarity score to consider for replacement
    Returns:
        Dictionary mapping skills to their replacements
    """
    replacements = {}

    for skill in similarity_df.index:
        for i in range(1, 11):  # Check all similar skills
            similar = similarity_df.loc[skill, f"similar_{i}"]
            score = similarity_df.loc[skill, f"score_{i}"]

            if pd.notna(similar) and pd.notna(score) and score >= similarity_threshold:
                # Choose the shorter one as the replacement
                if len(similar) < len(skill):
                    replacements[skill] = similar
                elif len(skill) < len(similar):
                    replacements[similar] = skill
    return replacements


# Apply replacements to each skill type
print("\nCreating and applying skill replacements...")

# Process soft skills
soft_skill_replacements = create_skill_replacements(soft_skills_df)
print("\nTop 10 Soft skill replacements:")
for old, new in list(soft_skill_replacements.items())[:10]:  # Show first 10 examples
    print(f"- '{old}' -> '{new}'")

# Process hard skills
hard_skill_replacements = create_skill_replacements(hard_skills_df)
print("\nTop 10 Hard skill replacements:")
for old, new in list(hard_skill_replacements.items())[:10]:  # Show first 10 examples
    print(f"- '{old}' -> '{new}'")

# Process fields of study
field_replacements = create_skill_replacements(field_skills_df)
print("\nTop 10 Field of study replacements:")
for old, new in list(field_replacements.items())[:10]:  # Show first 10 examples
    print(f"- '{old}' -> '{new}'")


# Function to replace skills in a list
def replace_skills(skills_list, replacements):
    return [replacements.get(skill, skill) for skill in skills_list]


# Apply replacements to the dataframe
print("\nApplying replacements to dataset...")
df["soft_skills"] = df["soft_skills"].apply(
    lambda x: replace_skills(x, soft_skill_replacements)
)
df["hard_skills"] = df["hard_skills"].apply(
    lambda x: replace_skills(x, hard_skill_replacements)
)
df["field_of_study"] = df["field_of_study"].apply(
    lambda x: replace_skills(x, field_replacements)
)

# Remove rows with NaN yearly_salary_midpoint
rows_before = len(df)
df = df.dropna(subset=["yearly_salary_midpoint"])
rows_removed = rows_before - len(df)
print(f"\nRemoved {rows_removed} rows with missing yearly salary midpoints")
print(f"Dataset reduced from {rows_before} to {len(df)} rows")

# Replace NaN values with "unspecified" for specific columns
columns_to_fill = [
    "seniority",
    "education_level",
    "contract_type",
    "location_flexibility",
]
df[columns_to_fill] = df[columns_to_fill].fillna("unspecified")

print("\nReplaced NaN values with 'unspecified' for:")
for col in columns_to_fill:
    print(f"- {col}: {df[col].value_counts()['unspecified']} replacements")

# Save final cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nFinal cleaned dataset saved to {OUTPUT_FILE}")

# Save replacement mappings for reference
replacement_df = pd.DataFrame(
    {
        "Type": ["Soft Skill"] * len(soft_skill_replacements)
        + ["Hard Skill"] * len(hard_skill_replacements)
        + ["Field of Study"] * len(field_replacements),
        "Original": list(soft_skill_replacements.keys())
        + list(hard_skill_replacements.keys())
        + list(field_replacements.keys()),
        "Replacement": list(soft_skill_replacements.values())
        + list(hard_skill_replacements.values())
        + list(field_replacements.values()),
    }
)
replacement_df.to_csv("skill_replacements.csv", index=False)
print(f"\nReplacement mappings saved to skill_replacements.csv")

# Generate profile report for final cleaned dataset
print("\nPreparing data for profile report...")
df_profile = df.copy()

# Convert list columns to string representation for profiling
list_columns = ["soft_skills", "hard_skills", "field_of_study"]
for col in list_columns:
    df_profile[col] = df_profile[col].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )

# Generate profile report
print("Generating profile report...")
profile = ProfileReport(
    df_profile,
    title="Final Cleaned Jobs Dataset Profile",
    explorative=True,
    minimal=False,
)
profile.to_file("final_cleaned_jobs_profile.html")
print("Profile report saved to final_cleaned_jobs_profile.html")

# Quick PyCaret model training and evaluation
print("\nPreparing data for PyCaret modeling...")

# Create modeling dataset
model_df = df.copy()

# Convert list columns to string for modeling
list_columns = ["soft_skills", "hard_skills", "field_of_study"]
for col in list_columns:
    model_df[col] = model_df[col].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )

# Initialize PyCaret regression setup
from pycaret.regression import *

# Setup the regression experiment
print("\nSetting up PyCaret regression experiment...")
reg_setup = setup(
    data=model_df,
    target="yearly_salary_midpoint",
    session_id=42,
    normalize=True,
    transformation=True,
    ignore_features=["country"],  # Ignore country as it's highly correlated with salary
    silent=True,
    use_gpu=True,  # Enable GPU if available
)

# Compare models
print("\nComparing different regression models...")
best_model = compare_models(n_select=3)

# Get the best model's performance metrics
print("\nDetailed evaluation of best model:")
evaluate_model(best_model)

# Create predictions
predictions = predict_model(best_model)
print("\nSample predictions vs actual values:")
print(predictions[["yearly_salary_midpoint", "prediction_label"]].head())

# Feature importance
print("\nFeature Importance Analysis:")
plot_model(best_model, plot="feature")

# Save the best model
save_model(best_model, "salary_prediction_model")
print("\nBest model saved as 'salary_prediction_model'")
