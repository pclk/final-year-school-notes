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
    # "paralegal",
    # "lawyer",
    # "attorney",
    # "legal secretary",
    # "data entry clerk",
    # "office manager",
    # "administrative assistant",
    # "customer service representative",
    # "executive assistant",
    # "receptionist",
    # "business development manager",
    # "management consultant",
    # "market research analyst",
    # "statistician",
    # "economist",
    # "ux researcher",
    # "technical writer",
    # "scientific writer",
    # "librarian",
    # "journalist",
    # "editor",
    # "translator",
    # "interpreter",
    # "pharmacovigilance specialist",
    # "clinical research associate",
    # "biostatistician",
    # "regulatory affairs specialist",
    # "lab technician",
    # "research associate",
    # "geneticist",
    # "zoologist",
    # "geologist",
    # "astronomer",
    # "mathematician",
    # "actuarial analyst",
    # "investment banker",
    # "portfolio manager",
    # "loan officer",
    # "risk manager",
    # "compliance officer",
    # "estate agent",
    # "insurance agent",
    # "real estate agent",
    # "social worker",
    # "psychologist",
    # "counselor",
    # "supply chain analyst",
    # "logistics manager",
    # "warehouse manager",
    # "inventory specialist",
    # "procurement specialist",
    # "planner",
    # "demand planner",
    # "merchandiser",
    # "quality control inspector",
    # "production supervisor",
    # "operations analyst",
    # "manufacturing supervisor",
    # "process engineer",
    # "industrial engineer",
    # "maintenance technician",
    # "facilities manager",
    # "construction worker",
    # "electrician",
    # "plumber",
    # "carpenter",
    # "welder",
    # "painter",
    # "hvac technician",
    # "automotive technician",
    # "diesel mechanic",
    # "aircraft mechanic",
    # "pilot",
    # "air traffic controller",
    # "flight attendant",
    # "customer success manager",
    # "sales director",
    # "account manager",
    # "brand manager",
    # "public relations manager",
    # "communications manager",
    # "content strategist",
    # "seo specialist",
    # "social media strategist",
    # "marketing analyst",
    # "advertising manager",
    # "media planner",
    # "event coordinator",
    # "market research manager",
    # "graphic artist",
    # "illustrator",
    # "video editor",
    # "photographer",
    # "game developer",
    # "animator",
    # "audio engineer",
    # "sound designer",
    # "film director",
    # "art director",
    # "fashion designer",
    # "interior designer",
    # "ux designer",
    # "ui designer",
    # "data visualization specialist",
    # "training coordinator",
    # "learning and development specialist",
    # "instructional technologist",
    # "corporate trainer",
    # "hr business partner",
    # "employee relations specialist",
    # "benefits administrator",
    # "compensation analyst",
    # "talent acquisition specialist",
    # "organizational development consultant",
    # "early childhood educator",
    # "special education teacher",
    # "curriculum developer",
    # "school administrator",
    # "academic advisor",
    # "education consultant",
    # "medical coder",
    # "medical biller",
    # "dental hygienist",
    # "dental assistant",
    # "medical technologist",
    # "surgical technician",
    # "radiologic technologist",
    # "pharmacy technician",
    # "healthcare administrator",
    # "clinical research coordinator",
    # "epidemiologist",
    # "dietitian",
    # "nutritionist",
    # "speech therapist",
    # "occupational therapist",
    # "athletic trainer",
    # "massage therapist",
    # "physical therapy assistant",
    # "legal secretary",
    # "court reporter",
    # "law clerk",
    # "litigation paralegal",
    # "contract administrator",
    # "compliance analyst",
    # "office assistant",
    # "clerical worker",
    # "data entry operator",
    # "bookkeeper",
    # "payroll clerk",
    # "administrative coordinator",
    # "project analyst",
    # "project controller",
    # "business development analyst",
    # "contract manager",
    # "consultant",
    # "management analyst",
    # "business intelligence analyst",
    # "research analyst",
    # "policy analyst",
    # "statistician analyst",
    # "actuarial consultant",
    # "investment analyst",
    # "financial consultant",
    # "portfolio analyst",
    # "tax analyst",
    # "credit analyst",
    # "underwriter",
    # "financial advisor",
    # "insurance underwriter",
    # "real estate appraiser",
    # "property manager",
    # "community manager",
    # "event manager",
    # "volunteer coordinator",
    # "case manager",
    # "mental health counselor",
    # "rehabilitation counselor",
    # "substance abuse counselor",
    # "probation officer",
    # "human services specialist",
    # "family support specialist",
    # "social media coordinator",
    # "copy editor",
    # "proofreader",
    # "grant writer",
    # "proposal writer",
    # "technical editor",
    # "content editor",
    # "news anchor",
    # "reporter",
    # "columnist",
    # "broadcaster",
    # "communication specialist",
    # "public affairs specialist",
    # "regulatory affairs manager",
    # "quality assurance specialist",
    # "clinical data manager",
    # "laboratory manager",
    # "research fellow",
    # "biochemist",
    # "microbiologist",
    # "environmental scientist",
    # "botanist",
    # "marine biologist",
    # "veterinarian",
    # "astrophysicist",
    # "geophysicist",
    # "data librarian",
    # "archivist",
    # "museum curator",
    # "academic librarian",
    # "investment consultant",
    # "risk analyst",
    # "fraud investigator",
    # "compliance specialist",
    # "real estate broker",
    # "customer service manager",
    # "customer support specialist",
    # "technical support specialist",
    # "renewable energy engineer",
    # "solar energy specialist",
    # "wind energy technician",
    # "energy auditor",
    # "sustainability manager",
    # "environmental consultant",
    # "waste management specialist",
    # "recycling coordinator",
    # "water treatment operator",
    # "environmental health specialist",
    # "agricultural engineer",
    # "horticulturist",
    # "farm manager",
    # "crop specialist",
    # "animal scientist",
    # "food scientist",
    # "chef de cuisine",
    # "sous chef",
    # "pastry chef",
    # "line cook",
    # "food and beverage manager",
    # "catering manager",
    # "sommelier",
    # "barista",
    # "restaurant host",
    # "hotel concierge",
    # "front desk agent",
    # "housekeeping supervisor",
    # "event marketing manager",
    # "tradeshow coordinator",
    # "conference planner",
    # "wedding planner",
    # "public relations coordinator",
    # "media relations manager",
    # "social media analyst",
    # "influencer marketing manager",
    # "digital marketing manager",
    # "email marketing specialist",
    # "paid search specialist",
    # "affiliate marketing manager",
    # "growth hacker",
    # "market development specialist",
    # "product marketing manager",
    # "brand strategist",
    # "creative director",
    # "digital designer",
    # "motion graphic designer",
    # "presentation designer",
    # "web content strategist",
    # "user interface designer",
    # "user experience architect",
    # "game designer",
    # "game artist",
    # "level designer",
    # "character animator",
    # "motion capture artist",
    # "technical artist",
    # "sound editor",
    # "music composer",
    # "film editor",
    # "film producer",
    # "set designer",
    # "costume designer",
    # "makeup artist",
    # "fashion stylist",
    # "interior architect",
    # "landscape architect",
    # "urban designer",
    # "town planner",
    # "community planner",
    # "transportation planner",
    # "construction estimator",
    # "construction superintendent",
    # "civil engineering technician",
    # "electrical engineering technician",
    # "mechanical engineering technician",
    # "draftsman",
    # "cad designer",
    # "building inspector",
    # "safety officer",
    # "occupational health specialist",
    # "quality engineer",
    # "reliability engineer",
    # "test engineer",
    # "validation engineer",
    # "chemical process technician",
    # "environmental technician",
    # "automation technician",
    # "robotics engineer",
    # "mechatronics engineer",
    # "aerospace engineer",
    # "materials engineer",
    # "bioprocess engineer",
    # "clinical laboratory scientist",
    # "medical imaging specialist",
    # "radiology technician",
    # "cardiovascular technologist",
    # "surgical assistant",
    # "patient care technician",
    # "medical transcriptionist",
    # "medical records specialist",
    # "healthcare consultant",
    # "public health specialist",
    # "health educator",
    # "recreational therapist",
    # "art therapist",
    # "music therapist",
    # "dance therapist",
    # "physical medicine specialist",
    # "medical geneticist",
    # "clinical pharmacist",
    # "research pharmacist",
    # "pharmaceutical sales representative",
    # "drug safety specialist",
    # "clinical data analyst",
    # "biostatistical programmer",
    # "epidemiological researcher",
    # "health policy analyst",
    # "community health worker",
    # "social work supervisor",
    # "case management director",
    # "child welfare specialist",
    # "family therapist",
    # "crisis counselor",
    # "rehabilitation specialist",
    # "veteran affairs counselor",
    # "substance abuse counselor",
    # "probation officer",
    # "parole officer",
    # "victim advocate",
    # "human resources director",
    # "talent management specialist",
    # "recruitment manager",
    # "training and development manager",
    # "hr generalist",
    # "compensation and benefits manager",
    # "payroll manager",
    # "employee relations manager",
    # "organizational development manager",
    # "learning technology specialist",
    # "e-learning designer",
    # "training specialist",
    # "instructional coach",
    # "academic dean",
    # "registrar",
    # "admissions counselor",
    # "student services coordinator",
    # "curriculum specialist",
    # "educational psychologist",
    # "school principal",
    # "school superintendent",
    # "university lecturer",
    # "professor emeritus",
    # "research scientist",
    # "postdoctoral researcher",
    # "laboratory assistant",
    # "laboratory technician",
    # "research associate",
    # "data curator",
    # "biomedical researcher",
    # "genetic counselor",
    # "genomics specialist",
    # "immunologist",
    # "neuroscientist",
    # "pathologist",
    # "pharmacologist",
    # "toxicologist",
    # "microbial geneticist",
    # "marine biologist",
    # "zoological park keeper",
    # "wildlife biologist",
    # "astronomy educator",
    # "geological surveyor",
    # "geospatial analyst",
    # "mathematician",
    # "statistical modeler",
    # "quantitative analyst",
    # "actuarial intern",
    # "investment banker analyst",
    # "financial planner",
    # "credit risk manager",
    # "financial risk analyst",
    # "portfolio manager assistant",
    # "hedge fund analyst",
    # "private equity analyst",
    # "venture capital analyst",
    # "compliance auditor",
    # "fraud examiner",
    # "internal auditor",
    # "tax preparer",
    # "accountant specialist",
    # "corporate controller",
    # "cfo",
    # "real estate portfolio manager",
    # "commercial real estate agent",
    # "lease administrator",
    # "property management assistant",
    # "community outreach coordinator",
    # "volunteer coordinator",
    # "nonprofit program manager",
    # "fundraising manager",
    # "development officer",
    # "public affairs manager",
    # "political analyst",
    # "lobbyist",
    # "policy maker",
    # "urban planner",
    # "technical writer manager",
    # "content strategist director",
    # "chief technology officer",
    # "data security manager",
    # "cybersecurity director",
    # "it project manager",
    # "systems engineer specialist",
    # "network administrator manager",
    # "database administrator director",
    # "software quality engineer",
    # "automation engineer manager",
    # "cloud solutions architect",
    # "ai researcher",
    # "machine learning specialist",
    # "deep learning engineer",
    # "blockchain security engineer",
    # "embedded systems developer",
    # "computer vision engineer",
    # "data governance specialist",
    # "data compliance analyst",
    # "information architect",
    # "knowledge manager",
    # "business transformation manager",
    # "change management specialist",
    # "process improvement specialist",
    # "agile coach",
    # "scrum team member",
    # "kanban manager",
    # "product growth manager",
    # "technical project manager",
    # "telecommunications engineer",
    # "broadcast engineer",
    # "satellite engineer",
    # "optical engineer",
    # "radar engineer",
    # "robotics technician",
    # "automation specialist",
    # "process control engineer",
    # "industrial designer",
    # "packaging engineer",
    # "logistics coordinator",
    # "supply chain director",
    # "warehouse supervisor",
    # "distribution manager",
    # "import/export manager",
    # "procurement manager",
    # "purchasing agent",
    # "inventory control manager",
    # "quality control director",
    # "compliance manager",
    # "safety manager",
    # "facility operations manager",
    # "maintenance manager",
    # "asset manager",
    # "vendor manager",
    # "contract negotiator",
    # "risk mitigation specialist",
    # "claims adjuster",
    # "loss prevention specialist",
    # "customer support director",
    # "customer relations specialist",
    # "technical trainer",
    # "cartographer",
    # "geographer",
    # "demographer",
    # "statistical geographer",
    # "remote sensing specialist",
    # "geographic information systems specialist",
    # "urban geographer",
    # "spatial analyst",
    # "data visualization developer",
    # "database architect",
    # "data modeling specialist",
    # "etl developer",
    # "business intelligence developer",
    # "data warehouse engineer",
    # "master data management specialist",
    # "data quality analyst",
    # "information security architect",
    # "penetration tester",
    # "vulnerability analyst",
    # "security operations center analyst",
    # "incident responder",
    # "cryptographer",
    # "ethical hacker",
    # "forensic analyst",
    # "network security engineer",
    # "cloud security specialist",
    # "biometrics specialist",
    # "quantum computing engineer",
    # "nanotechnology engineer",
    # "materials scientist",
    # "cognitive scientist",
    # "artificial intelligence researcher",
    # "machine vision specialist",
    # "natural language processing engineer",
    # "roboticist",
    # "automation consultant",
    # "systems integrator",
    # "devops consultant",
    # "site reliability manager",
    # "infrastructure architect",
    # "network operations center technician",
    # "telecommunications technician",
    # "voip engineer",
    # "wireless engineer",
    # "fiber optic technician",
    # "antenna engineer",
    # "broadcast technician",
    # "satellite communications engineer",
    # "optical communications engineer",
    # "radar systems engineer",
    # "sonar systems engineer",
    # "avionics technician",
    # "aircraft maintenance planner",
    # "airframe mechanic",
    # "powerplant mechanic",
    # "aerospace manufacturing engineer",
    # "spacecraft engineer",
    # "rocket scientist",
    # "astrobiologist",
    # "planetary scientist",
    # "solar physicist",
    # "cosmologist",
    # "theoretical physicist",
    # "particle physicist",
    # "nuclear engineer",
    # "radiological physicist",
    # "medical physicist",
    # "biophysicist",
    # "biomechanical engineer",
    # "ergonomist",
    # "rehabilitation engineer",
    # "prosthetist",
    # "orthotist",
    # "clinical trials manager",
    # "medical writer",
    # "scientific editor",
    # "scientific illustrator",
    # "medical illustrator",
    # "health informatics specialist",
    # "bioethicist",
    # "regulatory affairs director",
    # "pharmacovigilance manager",
    # "drug development manager",
    # "toxicology manager",
    # "clinical affairs manager",
    # "medical affairs director",
    # "managed care specialist",
    # "healthcare compliance officer",
    # "quality improvement manager",
    # "patient safety officer",
    # "health system administrator",
    # "long term care administrator",
    # "medical librarian",
    # "veterinary technician",
    # "zoological curator",
    "anthropolgist",
    "archaeologist",
    "paleontologist",
    "epidemologist",
    "virologist",
    "immunologist",
    "toxicologist",
    "pharmacologist",
    "neuroscientist",
    "cognitive scientist",
    "linguist",
    "etymologist",
    "philologist",
    "sociologist",
    "political scientist",
    "international relations specialist",
    "urban sociologist",
    "criminologist",
    "forensic scientist",
    "ballistics expert",
    "serologist",
    "dna analyst",
    "voice analyst",
    "handwriting analyst",
    "document examiner",
    "intelligence analyst",
    "counterintelligence analyst",
    "geospatial intelligence analyst",
    "signals intelligence analyst",
    "human intelligence collector",
    "open source intelligence analyst",
    "competitive intelligence analyst",
    "market intelligence analyst",
    "business intelligence consultant",
    "data governance manager",
    "data steward",
    "data quality engineer",
    "database developer",
    "database modeler",
    "etl architect",
    "data migration specialist",
    "cloud database administrator",
    "nosql database administrator",
    "big data engineer",
    "hadoop developer",
    "spark developer",
    "kafka engineer",
    "data streaming engineer",
    "data lake architect",
    "machine learning operations engineer",
    "artificial intelligence ethicist",
    "computer vision researcher",
    "natural language understanding engineer",
    "speech recognition engineer",
    "robotics software engineer",
    "autonomous systems engineer",
    "internet of things engineer",
    "edge computing engineer",
    "virtual reality developer",
    "augmented reality developer",
    "game programmer",
    "graphics programmer",
    "simulation engineer",
    "mathematical modeler",
    "statistical programmer",
    "biostatistician programmer",
    "econometrician",
    "financial engineer",
    "quantitative analyst developer",
    "algorithm developer",
    "cryptocurrency developer",
    "blockchain engineer",
    "smart contract developer",
    "decentralized application developer",
    "full stack javascript developer",
    "mern stack developer",
    "mean stack developer",
    "lamp stack developer",
    "rails developer",
    "django developer",
    "flask developer",
    "spring boot developer",
    ".net developer",
    "c++ developer",
    "c# developer",
    "golang developer",
    "rust developer",
    "scala developer",
    "kotlin developer",
    "swift developer",
    "objective-c developer",
    "php developer",
    "perl developer",
    "ruby on rails developer",
    "sql developer",
    "pl/sql developer",
    "t-sql developer",
    "database tester",
    "software test automation engineer",
    "performance test engineer",
    "security test engineer",
    "mobile test engineer",
    "embedded software engineer",
    "firmware engineer",
    "bios engineer",
    "driver developer",
    "operating systems engineer",
    "network security architect",
    "cloud security engineer",
    "application security engineer",
    "information security analyst",
    "security consultant",
    "security auditor",
    "security operations engineer",
    "devsecops engineer",
    "cloud architect",
    "solutions architect",
    "enterprise architect",
    "technical consultant",
    "it consultant",
    "sap consultant",
    "salesforce developer",
    "dynamics 365 consultant",
    "oracle fusion consultant",
    "workday consultant",
    "servicenow developer",
    "cybersecurity consultant",
    "risk consultant",
    "compliance consultant",
    "management consultant",
    "strategy consultant",
    "operations consultant",
    "human capital consultant",
    "financial consultant",
    "tax consultant",
    "audit consultant",
    "actuarial consultant",
    "investment consultant",
    "real estate consultant",
    "environmental consultant",
    "sustainability consultant",
    "supply chain consultant",
    "logistics consultant",
    "healthcare consultant",
    "education consultant",
    "nonprofit consultant",
    "public sector consultant",
    "marketing consultant",
    "sales consultant",
    "hr consultant",
    "training consultant",
    "legal consultant",
    "technical recruiter",
    "executive recruiter",
    "headhunter",
    "talent acquisition manager",
    "recruiting coordinator",
    "hr business partner director",
    "compensation and benefits director",
    "training and development director",
    "organizational development director",
    "employee relations director",
    "hr director",
    "chief human resources officer",
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
