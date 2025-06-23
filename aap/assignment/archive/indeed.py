from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re
from random import uniform
import csv
import os


indeed_links_map = {
    "US": "https://www.indeed.com/jobs?q={query}",
    "SG": "https://sg.indeed.com/jobs?q={query}",
    "IN": "https://in.indeed.com/jobs?q={query}",
}


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")

    # Add user agent
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Add additional Chrome options to make it more stable
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Set page load timeout
    driver.set_page_load_timeout(30)

    return driver


# --- Helper Functions for Extraction ---


def extract_job_title(text):
    # Usually the first line or comes before "- job post"
    match = re.search(r"^(.*?)(?=\n- job post)", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    match = re.search(r"^(.*?)(?=\n-)", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def extract_company(text):
    match = re.search(r"- job post\n(.*?)\n", text)
    if match:
        company = match.group(1).strip()
        if company.count(" ") > 4:
            return None
        return company
    return None


def extract_company_rating(text):
    match = re.search(r"(\d+\.\d+)\s+out of 5 stars", text)
    if match:
        return match.group(1).strip()
    return None


def extract_location(text):
    match = re.search(
        r"Location:\s*(.*?)\n|(\w+(?:,\s*\w+)*)•?Remote|(\w+(?:,\s*\w+)*)(?=-)", text
    )
    if match:
        return match.group(1) or match.group(2) or match.group(3)
    return None


def extract_job_type(text):
    match = re.search(r"Job type\n(.*?)\n", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"(Full-time|Part-time|Contract|Temporary)", text)
    if match:
        return match.group(1).strip()
    return None


def extract_work_setting(text):
    match = re.search(r"Work setting\n(.*?)\n", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"(Remote|Hybrid|On-site)", text)
    if match:
        return match.group(1).strip()
    return None


def extract_salary(text):
    match = re.search(
        r"(\$\d+,\d+|\$\d+\s*-\s*\$\d+,\d+|\$\d+\s*-\s*\$\d+|\$\d+ an hour|\$\d+,\d+ an hour)",
        text,
    )
    if match:
        return match.group(1).strip()
    return None


def scrape_indeed_jobs(query, country, num_pages=1):
    if country not in indeed_links_map:
        raise ValueError(
            f"Country '{country}' not supported. Available countries: {', '.join(indeed_links_map.keys())}"
        )

    driver = setup_driver()
    jobs = []
    base_url = indeed_links_map[country]

    # Define CSV headers for structured data
    fieldnames = [
        "Job Title",
        "Country",
        "Company",
        "Company Rating",
        "Job Type",
        "Work Setting",
        "Salary/Pay Rate",
        "Job Details",
    ]

    # Create or open CSV file
    file_exists = os.path.isfile("indeed.csv")
    with open("indeed.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    try:
        for page in range(num_pages):
            # Construct URL for each page using the country-specific base URL
            if page == 0:
                url = base_url.format(query=query)
            else:
                url = f"{base_url.format(query=query)}&start={page*10}"

            print(f"Scraping page {page + 1}: {url}")
            driver.get(url)

            # Wait for job cards to load
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "cardOutline"))
                )
                # Scroll to load all content
                driver.execute_script("""
                                function smoothScroll() {
                                    let scrollHeight = document.body.scrollHeight;
                                    let currentPosition = 0;
                                    let step = 8;
                                    
                                    function scroll() {
                                        if (currentPosition < scrollHeight) {
                                            currentPosition = Math.min(currentPosition + step, scrollHeight);
                                            window.scrollTo(0, currentPosition);
                                            setTimeout(scroll, 100);
                                        }
                                    }
                                    scroll();
                                }
                                smoothScroll();
                                """)
                time.sleep(2)
            except TimeoutException:
                print(f"Timeout waiting for page {page + 1} to load")
                continue

            # Get all job cards on the page
            job_cards = driver.find_elements(By.CLASS_NAME, "cardOutline")

            if not job_cards:
                print(f"No jobs found on page {page + 1}")
                continue

            # Process each job card
            for i, card in enumerate(job_cards):
                try:
                    print(f"Processing job {i+1} of {len(job_cards)} on page {page+1}")
                    job = {}

                    # Click on the job card to load details
                    try:
                        # Find and click the job title link
                        job_title = card.find_element(By.CLASS_NAME, "jcs-JobTitle")
                        driver.execute_script("arguments[0].click();", job_title)

                        # Wait for job details to load
                        time.sleep(uniform(2, 3))

                        # Wait for the job details panel
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located(
                                (By.CLASS_NAME, "jobsearch-JobComponent")
                            )
                        )
                    except Exception as e:
                        print(f"Error clicking job card: {str(e)}")
                        continue

                    # Extract detailed information from the job details panel
                    try:
                        job_details = driver.find_element(
                            By.CLASS_NAME, "jobsearch-JobComponent"
                        )

                        # get text of whole job details
                        try:
                            job["job_details"] = job_details.text.strip()
                        except:
                            pass

                    except Exception as e:
                        print(f"Error extracting job details: {str(e)}")

                    if job:  # Only append if we found some data
                        jobs.append(job)
                        # Extract structured data from job details
                        job_text = job.get("job_details", "")

                        csv_job = {
                            "Job Title": extract_job_title(job_text),
                            "Country": country,  # Add the country information
                            "Company": extract_company(job_text),
                            "Company Rating": extract_company_rating(job_text),
                            "Location": extract_location(job_text),
                            "Job Type": extract_job_type(job_text),
                            "Work Setting": extract_work_setting(job_text),
                            "Salary/Pay Rate": extract_salary(job_text),
                            "Job Details": job_text,
                        }

                        # Save to CSV immediately after processing each job
                        with open(
                            "indeed.csv", mode="a", newline="", encoding="utf-8"
                        ) as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow(csv_job)

                    # Random delay between processing cards
                    time.sleep(uniform(1, 2))

                except Exception as e:
                    print(f"Error processing job card: {str(e)}")
                    continue

            print(f"Found {len(job_cards)} jobs on page {page + 1}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        try:
            driver.quit()  # Always close the browser
        except:
            pass

    return jobs


# Iterate over each country in the indeed_links_map
for country in indeed_links_map.keys():
    print(f"\nStarting job search for {country}...")
    try:
        jobs = scrape_indeed_jobs("python developer", country, num_pages=1)
        print(f"Successfully completed scraping for {country}")
    except Exception as e:
        print(f"Error scraping jobs for {country}: {str(e)}")
        continue
