from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import csv
import os
import json
from random import uniform
from selenium.webdriver.chrome.options import Options
from datetime import datetime


def setup_driver():
    """Configure and return the Chrome WebDriver with appropriate options."""
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    driver = webdriver.Chrome(options=chrome_options)
    return driver


def extract_job_details(soup, driver, job_card):
    try:
        job_data = {
            "country": "",
            "job_description": "",
            "location": "",
            "salary": "",
            "job_title": "",
            "job_link": "",
        }

        # Get the job link and store it
        job_link = job_card.find("a", class_="JobCard_trackingLink__HMyun")
        if job_link and job_link.get("href"):
            href = job_link.get("href")
            if href.startswith("/"):
                base_url = "https://www.glassdoor.com"
                if "glassdoor.sg" in driver.current_url:
                    base_url = "https://www.glassdoor.sg"
                elif "glassdoor.co.in" in driver.current_url:
                    base_url = "https://www.glassdoor.co.in"
                job_data["job_link"] = base_url + href
            else:
                job_data["job_link"] = href

        # Extract salary by iterating through possible salary elements
        try:
            # Find all salary elements in the job cards
            salary_elements = driver.find_elements(
                By.CSS_SELECTOR, "[data-test='detailSalary']"
            )

            # Get the current job's href to match with the correct salary
            current_job_href = job_link.get("href") if job_link else None

            if current_job_href:
                # Find the matching salary element for this job card
                salary_found = False
                for salary_element in salary_elements:
                    # Get the parent job card element
                    parent_card = salary_element.find_element(
                        By.XPATH, ("./ancestor::div[contains(@class, 'jobCard')]")
                    )
                    card_link = parent_card.find_element(
                        By.CSS_SELECTOR, "a[data-test='job-link']"
                    )

                    # Check if this is the salary for our current job
                    if card_link.get_attribute("href").endswith(current_job_href):
                        salary_text = salary_element.text.strip()
                        if salary_text and salary_text != "Salary not available":
                            job_data["salary"] = salary_text
                            salary_found = True
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Found matching salary: {job_data['salary']}"
                            )
                            break

                if not salary_found:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] No salary found, skipping job"
                    )
                    return None
            else:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] No job link found, skipping job"
                )
                return None

        except Exception as e:
            print(f"Error getting salary: {str(e)}")
            return None

        # Find and click the job title link
        try:
            # Check if this job's details are already visible
            current_url = driver.current_url
            job_href = job_link.get("href")

            # Only skip the click if this specific job is already showing
            if job_href in current_url:
                print("This job's details already visible, skipping click")
            else:
                job_title_link = WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located(
                        (
                            By.CSS_SELECTOR,
                            f"a[data-test='job-link'][href='{job_href}']",
                        )
                    )
                )

                # Scroll the element into view
                driver.execute_script(
                    "arguments[0].scrollIntoView(true);", job_title_link
                )
                time.sleep(uniform(0.05, 0.1))  # Minimal pause after scrolling

                # Click using JavaScript
                driver.execute_script("arguments[0].click();", job_title_link)

                # No need for extra wait here since we'll wait for description element in get_full_description
        except Exception as e:
            print(f"Error clicking job details: {str(e)}")
            return None

        # Now extract the job details from the expanded view
        job_data["job_description"] = (
            get_full_description(driver) or "Description not available"
        )

        # Extract location using XPath
        try:
            location_element = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//*[@id='app-navigation']/div[4]/div[2]/div[2]/div/div[1]/header/div[1]/div",
                    )
                )
            )
            job_data["location"] = location_element.text.strip()
        except Exception as e:
            print(f"Error getting location: {str(e)}")
            job_data["location"] = "Location not available"

        # Extract job title by iterating through possible title elements
        try:
            # Find all title elements in the job cards
            title_elements = driver.find_elements(
                By.XPATH,
                "//*[@id='left-column']/div[2]/ul/li/div/div/div[1]/div[1]/a[1]",
            )

            # Get the current job's href to match with the correct title
            current_job_href = job_link.get("href") if job_link else None

            if current_job_href:
                # Find the matching title element for this job card
                for title_element in title_elements:
                    # Get the parent job card element
                    parent_card = title_element.find_element(
                        By.XPATH, "./ancestor::div[contains(@class, 'jobCard')]"
                    )
                    card_link = parent_card.find_element(
                        By.CSS_SELECTOR, "a[data-test='job-link']"
                    )

                    # Check if this is the title for our current job
                    if card_link.get_attribute("href").endswith(current_job_href):
                        job_data["job_title"] = title_element.text.strip()
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Found matching job title: {job_data['job_title']}"
                        )
                        break
                else:
                    job_data["job_title"] = "Title not available"
            else:
                job_data["job_title"] = "Title not available"

        except Exception as e:
            print(f"Error getting job title: {str(e)}")
            job_data["job_title"] = "Title not available"

        # Store the job link for reference
        job_link = job_card.find("a", class_="JobCard_trackingLink__HMyun")
        if job_link and job_link.get("href"):
            href = job_link.get("href")
            if href.startswith("/"):
                base_url = "https://www.glassdoor.com"
                if "glassdoor.sg" in driver.current_url:
                    base_url = "https://www.glassdoor.sg"
                elif "glassdoor.co.in" in driver.current_url:
                    base_url = "https://www.glassdoor.co.in"
                job_data["job_link"] = base_url + href
            else:
                job_data["job_link"] = href

        return job_data

    except Exception as e:
        print(f"Error extracting job details: {str(e)}")
        return None


def get_full_description(driver):
    """Get the full job description from the expanded job card view."""
    try:
        # First ensure no modal is present
        def close_any_modal():
            try:
                close_button = WebDriverWait(driver, 0.5).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "CloseButton"))
                )
                if close_button.is_displayed():
                    driver.execute_script("arguments[0].click();", close_button)
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Modal detected and closed"
                    )
                    time.sleep(uniform(0.05, 0.1))
                    return True
                return False
            except:
                return False

        # Try to close modal if present
        close_any_modal()

        # Use the most reliable Show More button selector
        try:
            show_more_button = WebDriverWait(driver, 0.5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "button[class*='JobDetails_showMore___']")
                )
            )
            driver.execute_script("arguments[0].click();", show_more_button)
            print(
                f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Clicked 'Show More' button"
            )
        except Exception as e:
            print("'Show More' button not found")

        # Try to get the description using the specific XPath
        try:
            description_element = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//*[@id='app-navigation']/div[4]/div[2]/div[2]/div/div[1]/section/div[2]/div[1]",
                    )
                )
            )
            description = description_element.text.strip()
            if description:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Found job description"
                )
                return description
        except Exception as e:
            print(f"Error getting description using XPath: {str(e)}")

        return "Description not available"

    except Exception as e:
        print(f"Error getting full description: {str(e)}")
        return "Description not available"


def extract_job_id(url):
    """Extract jobListingId from URL"""
    try:
        if "?" in url:
            params = dict(param.split("=") for param in url.split("?")[1].split("&"))
            return params.get("jobListingId")
    except:
        return None
    return None


def load_metadata():
    """Load metadata of previously scraped jobs"""
    if os.path.exists("scraping_metadata.json"):
        with open("scraping_metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert the URLs to job IDs
            job_ids = {
                extract_job_id(url)
                for url in data["scraped_jobs"]
                if extract_job_id(url)
            }
            data["scraped_jobs"] = job_ids
            return data
    return {"scraped_jobs": set(), "last_scrape_date": None, "total_jobs_scraped": 0}


def save_metadata(metadata):
    """Save metadata of scraped jobs"""
    # Convert set to list for JSON serialization
    metadata["scraped_jobs"] = list(metadata["scraped_jobs"])
    with open("scraping_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def scrape_glassdoor_jobs(query, country, jobs_per_country=50):
    """Main function to scrape Glassdoor jobs.

    Args:
        query (str): Job search query
        country (str): Country code to search in
        jobs_per_country (int): Number of jobs to scrape per country
    """
    if country not in glassdoor_links_map:
        raise ValueError(
            f"Country '{country}' not supported. Available countries: {', '.join(glassdoor_links_map.keys())}"
        )

    # Load metadata of previously scraped jobs
    metadata = load_metadata()
    scraped_jobs = set(metadata["scraped_jobs"])  # Convert back to set
    print(f"Found {len(scraped_jobs)} previously scraped jobs")
    print("First few scraped job links:", list(scraped_jobs)[:3])  # Debug print

    driver = setup_driver()
    jobs = []

    # Define CSV headers
    fieldnames = [
        "query",
        "country",
        "job_description",
        "location",
        "salary",
        "job_title",
        "job_link",
    ]

    # Create or open CSV file
    file_exists = os.path.isfile("glassdoor.csv")
    with open("glassdoor.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    try:
        base_url = glassdoor_links_map[country]
        query_formatted = query.replace(" ", "-")
        # Calculate the correct query length based on the actual position in the URL
        query_len = 14
        if country == "US":
            query_len = len(query_formatted) + 14
        elif country == "SG":
            query_len = len(query_formatted) + 10
        elif country == "IN":
            query_len = len(query_formatted) + 6

        url = base_url.format(query=query_formatted, query_len=query_len)

        current_job_count = 0
        page = 1

        while current_job_count < jobs_per_country:
            if page > 1:
                url = f"{url}?p={page}"
            print(
                f"\nProcessing page {page} for {country}. Current jobs: {current_job_count}/{jobs_per_country}"
            )

            driver.get(url)
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Current URL: {url}"
            )
            print(f"Attempting to load job cards on page {page + 1}...")

            try:
                # Single wait for job cards to load using a specific selector
                WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "[data-test='job-link']")
                    )
                )
                print(
                    f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Job cards loaded successfully"
                )
            except TimeoutException:
                print("ERROR: Timeout waiting for job cards to load")
                print("Current page source:")
                print(driver.page_source[:500])  # Print first 500 chars of page source
                raise

            # Parse the page
            print("Parsing page with BeautifulSoup...")
            soup = BeautifulSoup(driver.page_source, "html.parser")
            job_cards = soup.find_all("div", class_="jobCard")
            print(
                f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Found {len(job_cards)} job cards on the page"
            )

            for job_card in job_cards:
                # Get job link before full extraction to check if already scraped
                job_link = job_card.find("a", class_="JobCard_trackingLink__HMyun")
                if job_link and job_link.get("href"):
                    href = job_link.get("href")

                    if href.startswith("/"):
                        base_url = "https://www.glassdoor.com"
                        if "glassdoor.sg" in driver.current_url:
                            base_url = "https://www.glassdoor.sg"
                        elif "glassdoor.co.in" in driver.current_url:
                            base_url = "https://www.glassdoor.co.in"
                        full_job_link = base_url + href
                    else:
                        full_job_link = href

                    # Extract jobListingId from the current job link
                    current_job_id = extract_job_id(full_job_link)
                    if current_job_id and current_job_id in scraped_jobs:
                        # Convert set to list for indexing
                        scraped_list = list(scraped_jobs)
                        index = scraped_list.index(current_job_id)
                        print(
                            f"Skipping already scraped job (index {index}): jobListingId={current_job_id}"
                        )
                        continue

                job = extract_job_details(soup, driver, job_card)
                if job:
                    # Add country and search term information
                    job["country"] = country
                    job["query"] = query

                    # Job description is already fetched in extract_job_details
                    if not job.get("job_description"):
                        job["job_description"] = "Description not available"

                    # Add job ID to scraped jobs set
                    job_id = extract_job_id(job["job_link"])
                    if job_id:
                        scraped_jobs.add(job_id)

                    # Save to CSV
                    with open(
                        "glassdoor.csv", mode="a", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(job)

                    jobs.append(job)
                    current_job_count += 1

                    # Check if we've reached the target number of jobs
                    if current_job_count >= jobs_per_country:
                        print(
                            f"Reached target of {jobs_per_country} jobs for {country}"
                        )
                        break

            # Break the loop if we've reached the target
            if current_job_count >= jobs_per_country:
                break

            # After processing all job cards, try to click "Show More Jobs" button
            try:
                # Quick check for modal
                try:
                    close_button = driver.find_element(By.CLASS_NAME, "CloseButton")
                    if close_button.is_displayed():
                        driver.execute_script("arguments[0].click();", close_button)
                except:
                    pass  # No modal present, continue

                show_more_jobs = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '//*[@id="left-column"]/div[2]/div/div/button')
                    )
                )

                # Scroll the button into view and ensure it's clickable
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", show_more_jobs
                )
                WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '//*[@id="left-column"]/div[2]/div/div/button')
                    )
                )

                # Click using JavaScript
                driver.execute_script("arguments[0].click();", show_more_jobs)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Clicked 'Show More Jobs' button"
                )

                # Wait for new job cards to load (wait for count to increase)
                old_count = len(driver.find_elements(By.CLASS_NAME, "jobCard"))
                try:
                    WebDriverWait(driver, 0.5).until(
                        lambda x: len(x.find_elements(By.CLASS_NAME, "jobCard"))
                        > old_count
                    )
                except TimeoutException:
                    print("Timeout waiting for new cards, continuing anyway...")

                # Update the soup and job cards with new content
                soup = BeautifulSoup(driver.page_source, "html.parser")
                job_cards = soup.find_all("div", class_="jobCard")
                print(
                    f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Found {len(job_cards)} job cards after loading more"
                )

            except Exception as e:
                print(
                    f"No more jobs to load or error clicking 'Show More Jobs' button: {str(e)}"
                )
                break  # Exit the loop if we can't load more jobs

    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    finally:
        # Update metadata before quitting
        metadata["scraped_jobs"] = list(scraped_jobs)
        metadata["last_scrape_date"] = datetime.now().isoformat()
        metadata["total_jobs_scraped"] = len(scraped_jobs)
        save_metadata(metadata)

        driver.quit()

    return jobs


# Common job search terms
COMMON_SEARCH_TERMS = [
    # "software engineer",
    # "data scientist",
    # "product manager",
    # "data analyst",
    # "software developer",
    # "project manager",
    # "business analyst",
    # "full stack developer",
    # "data engineer",
    # "frontend developer",
    # "backend developer",
    # "devops engineer",
    # "machine learning engineer",
    # "systems engineer",
    # "qa engineer",
    # "cloud engineer",
    # "java developer",
    # "python developer",
    # "web developer",
    # "solutions architect",
    # "it manager",
    # "network engineer",
    # "security engineer",
    # "database administrator",
    # "ui ux designer",
    # "scrum master",
    # "android developer",
    # "ios developer",
    # "site reliability engineer",
    # "technical lead",
    # "automation engineer",
    # "research scientist",
    # "ai engineer",
    # "blockchain developer",
    # "cloud architect",
    # "cybersecurity analyst",
    # "data architect",
    # "embedded engineer",
    # "full stack engineer",
    # "infrastructure engineer",
    # "javascript developer",
    # "mobile developer",
    # "network administrator",
    # "product owner",
    # "quality assurance",
    # "ruby developer",
    # "security analyst",
    # "software architect",
    # "systems administrator",
    # "technical architect",
    # "unity developer",
    # "accountant",
    # "financial analyst",
    # "auditor",
    # "financial manager",
    # "actuary",
    # "marketing manager",
    # "marketing specialist",
    # "sales manager",
    # "sales representative",
    # "digital marketing specialist",
    # "graphic designer",
    # "copywriter",
    # "content writer",
    # "public relations specialist",
    # "social media manager",
    # "human resources manager",
    # "hr specialist",
    # "recruiter",
    # "training manager",
    # "payroll specialist",
    # "teacher",
    # "professor",
    # "instructional designer",
    # "principal",
    # "school counselor",
    # "nurse",
    # "physician",
    # "pharmacist",
    # "medical assistant",
    # "physical therapist",
    # "registered nurse",
    # "medical doctor",
    # "therapist",
    # "project coordinator",
    # "operations manager",
    # "supply chain manager",
    # "logistics coordinator",
    # "purchasing manager",
    # "restaurant manager",
    # "chef",
    # "bartender",
    # "waiter/waitress",
    # "event planner",
    # "hotel manager",
    # "civil engineer",
    # "electrical engineer",
    # "mechanical engineer",
    # "chemical engineer",
    # "environmental engineer",
    # "architect",
    # "urban planner",
    # "construction manager",
    # "biomedical engineer",
    # "manufacturing engineer",
    # "legal assistant",
    "paralegal",
    "lawyer",
    "attorney",
    "legal secretary",
    "data entry clerk",
    "office manager",
    "administrative assistant",
    "customer service representative",
    "executive assistant",
    "receptionist",
    "business development manager",
    "management consultant",
    "market research analyst",
    "statistician",
    "economist",
    "ux researcher",
    "technical writer",
    "scientific writer",
    "librarian",
    "journalist",
    "editor",
    "translator",
    "interpreter",
    "pharmacovigilance specialist",
    "clinical research associate",
    "biostatistician",
    "regulatory affairs specialist",
    "lab technician",
    "research associate",
    "geneticist",
    "zoologist",
    "geologist",
    "astronomer",
    "mathematician",
    "actuarial analyst",
    "investment banker",
    "portfolio manager",
    "loan officer",
    "risk manager",
    "compliance officer",
    "estate agent",
    "insurance agent",
    "real estate agent",
    "social worker",
    "psychologist",
    "counselor",
]


# Glassdoor country-specific URLs
glassdoor_links_map = {
    "US": "https://www.glassdoor.com/Job/united-states-{query}-jobs-SRCH_IL.0,13_IN1_KO14,{query_len}.htm",
    "SG": "https://www.glassdoor.sg/Job/singapore-{query}-jobs-SRCH_IL.0,9_IN217_KO10,{query_len}.htm",
    "IN": "https://www.glassdoor.co.in/Job/india-{query}-jobs-SRCH_IL.0,5_IN115_KO6,{query_len}.htm?includeNoSalaryJobs=true",
}

if __name__ == "__main__":
    # Configuration
    JOBS_PER_COUNTRY = 30

    # Create a log file for the entire scraping session
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(
        f"scraping_session_{session_timestamp}.log", "w", encoding="utf-8"
    ) as log_file:
        # Iterate through each search term
        for search_term in COMMON_SEARCH_TERMS:
            log_file.write(
                f"\n{'='*50}\nProcessing search term: {search_term}\n{'='*50}\n"
            )
            print(f"\n{'='*50}\nProcessing search term: {search_term}\n{'='*50}")

            # Iterate through each country
            for country in glassdoor_links_map.keys():
                message = (
                    f"\nScraping Glassdoor jobs for {country} - Search: {search_term}"
                )
                print(message)
                log_file.write(message + "\n")

                try:
                    jobs = scrape_glassdoor_jobs(
                        search_term, country, jobs_per_country=JOBS_PER_COUNTRY
                    )

                    message = f"Successfully scraped {len(jobs)} jobs from {country} for {search_term}"
                    print(message)
                    log_file.write(message + "\n")

                except Exception as e:
                    error_message = (
                        f"Error scraping {country} for {search_term}: {str(e)}"
                    )
                    print(error_message)
                    log_file.write(error_message + "\n")

                # Add a minimal delay between countries
                time.sleep(uniform(0.1, 0.5))

            # Add a minimal delay between search terms
            time.sleep(uniform(0.1, 0.5))
# |%%--%%| <0|0>
import pandas as pd
from anthropic import Anthropic
import tiktoken
import json
from typing import Dict, Tuple

import numpy as np


def analyze_glassdoor_data():
    # Read the CSV file
    print("\nReading glassdoor.csv...")
    df = pd.read_csv("glassdoor.csv")

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

    # Check for very short or empty descriptions
    short_desc = df[df["job_description"].str.len() < 100]
    print(f"\nJobs with very short descriptions (<100 chars): {len(short_desc)}")

    # Check for invalid salaries (if they don't contain numbers)
    invalid_salaries = df[~df["salary"].str.contains(r"\d", na=True)]
    print(f"Jobs with potentially invalid salaries: {len(invalid_salaries)}")

    # Check for unusual locations
    print("\nUnique locations found:")
    print(df["location"].value_counts().head(10))

    # Save problematic entries to a separate CSV for review
    problematic = df[
        (df.isnull().any(axis=1))  # Any missing values
        | (df.duplicated())  # Duplicates
        | (df["job_description"].str.len() < 100)  # Short descriptions
        | (~df["salary"].str.contains(r"\d", na=True))  # Invalid salaries
    ]

    if len(problematic) > 0:
        problematic.to_csv("problematic_entries.csv", index=False)
        print(
            f"\nSaved {len(problematic)} problematic entries to 'problematic_entries.csv'"
        )


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")  # Claude's encoding
    return len(encoding.encode(text))


def calculate_claude_cost(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    """Calculate Claude API cost based on token usage"""
    input_cost_per_1k = 0.015
    output_cost_per_1k = 0.075

    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(total_cost, 4),
    }


def analyze_token_usage():
    """Analyze token usage and cost for EDA processing"""
    print("\n=== TOKEN USAGE ANALYSIS ===")

    # Read the CSV file
    df = pd.read_csv("glassdoor.csv")

    total_input_tokens = 0
    total_output_tokens = 0

    # Template tokens (counted once)
    template = """Analyze this job posting and extract the following features in JSON format:
        - soft_skills: List of soft skills mentioned (communication, leadership, etc)
        - hard_skills: List of technical skills, tools, languages required
        - location_flexibility: One of [remote, hybrid, onsite, unspecified]
        - contract_type: One of [full-time, part-time, contract, internship, unspecified] 
        - education_level: Minimum required education level [high_school, bachelors, masters, phd, unspecified]
        - field_of_study: Required field of study or major
        - min_years_experience: Minimum years of experience required (numeric, -1 if unspecified)
        - salary_range: Extract salary range if available [min, max, currency, period(yearly/monthly/hourly)]"""

    template_tokens = count_tokens(template)
    print(f"\nTemplate tokens (per request): {template_tokens}")

    # Analyze each job
    for _, row in df.iterrows():
        prompt_text = f"""
        Job Title: {row['job_title']}
        Location: {row['location']}
        Salary: {row['salary']}
        Description: {row['job_description']}
        """

        input_tokens = template_tokens + count_tokens(prompt_text)
        total_input_tokens += input_tokens

        # Estimate output tokens based on typical JSON response
        sample_output = {
            "soft_skills": ["communication", "teamwork"],
            "hard_skills": ["python", "sql"],
            "location_flexibility": "remote",
            "contract_type": "full-time",
            "education_level": "bachelors",
            "field_of_study": "computer science",
            "min_years_experience": 3,
            "salary_range": {
                "min": 80000,
                "max": 120000,
                "currency": "USD",
                "period": "yearly",
            },
        }
        output_tokens = count_tokens(json.dumps(sample_output))
        total_output_tokens += output_tokens

    # Calculate total cost
    cost_analysis = calculate_claude_cost(total_input_tokens, total_output_tokens)

    print("\n=== TOTAL TOKEN USAGE AND COST ===")
    print(f"Total Input Tokens: {cost_analysis['input_tokens']}")
    print(f"Total Output Tokens: {cost_analysis['output_tokens']}")
    print(f"Input Cost: ${cost_analysis['input_cost']}")
    print(f"Output Cost: ${cost_analysis['output_cost']}")
    print(f"Total Cost: ${cost_analysis['total_cost']}")


if __name__ == "__main__":
    analyze_glassdoor_data()
    analyze_token_usage()
# |%%--%%| <0|0>

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

            # Clean and extract JSON from response
            response_text = response.text.strip()

            # Debug print to see what we're getting
            # print("Raw response:", response_text)

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
                    features = json.loads(json_str)
                else:
                    raise Exception("Could not find valid JSON in response")

            return features

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return {
                "soft_skills": [],
                "hard_skills": [],
                "location_flexibility": "unspecified",
                "contract_type": "unspecified",
                "education_level": "unspecified",
                "field_of_study": "unspecified",
                "min_years_experience": -1,
            }


def generate_features():
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
        print("No existing eda.csv found. Starting from beginning.")
        df = glassdoor_df
        existing_eda_df = None

    # Initialize feature extractor
    extractor = JobFeatureExtractor()

    # Extract features for each job
    features = []
    total_rows = len(df)
    for _, row in tqdm(df.iterrows(), desc="Extracting features", total=total_rows):
        prompt_text = f"""
        Job Title: {row['job_title']}
        Location: {row['location']}
        Salary: {row['salary']}
        Description: {row['job_description']}
        """
        features.append(extractor.extract_features(prompt_text))

    # Convert features to DataFrame
    features_df = pd.DataFrame(features)

    # Combine with original DataFrame for new rows
    new_result_df = pd.concat([df, features_df], axis=1)

    # Combine with existing results if they exist
    if existing_eda_df is not None:
        result_df = pd.concat([existing_eda_df, new_result_df], axis=0)
    else:
        result_df = new_result_df

    # Save to CSV without index
    result_df.to_csv("eda.csv", index=False)

    # Print some basic statistics
    print("\nFeature Extraction Complete!")
    print(f"Total jobs processed: {len(result_df)}")
    print("\nLocation Flexibility Distribution:")
    print(result_df["location_flexibility"].value_counts())
    print("\nContract Type Distribution:")
    print(result_df["contract_type"].value_counts())
    print("\nEducation Level Distribution:")
    print(result_df["education_level"].value_counts())
    print(
        "\nAverage Years of Experience Required:",
        result_df[result_df["min_years_experience"] != -1][
            "min_years_experience"
        ].mean(),
    )


generate_features()
# |%%--%%| <0|0>

from ydata_profiling import ProfileReport
import warnings

warnings.filterwarnings("ignore")


def generate_profile_report():
    print("Reading eda.csv file...")
    df = pd.read_csv("eda.csv")

    print("Generating profile report...")
    profile = ProfileReport(
        df,
        title="Glassdoor Jobs Analysis Report",
        explorative=True,
        dark_mode=True,
        correlations={
            "auto": {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        },
    )

    print("Saving report to jobs_analysis_report.html...")
    profile.to_file("jobs_analysis_report.html")
    print("Report generation complete!")


if __name__ == "__main__":
    generate_profile_report()
