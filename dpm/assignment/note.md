
# WeAreDoctors Healthcare System Migration to NoSQL Database: A Case Study

WeAreDoctors is a well-established healthcare service company operating globally, providing round-the-clock healthcare services to its diverse customer base. Initially conceived as a startup, WeAreDoctors pioneered online medical consultations via its web portal, relying on a relational database to manage its healthcare service system.  However, with substantial business growth and geographical expansion, the company's IT team encountered increasing challenges in maintaining their existing relational database infrastructure while ensuring consistently reliable online consultation services. To address these scalability and performance concerns, WeAreDoctors has decided to migrate their current healthcare system to a NoSQL database solution, specifically recognizing the proven advantages of NoSQL databases in terms of high performance, scalability, and fault tolerance.  In light of this strategic shift, your expertise has been sought to facilitate the transformation of their existing system utilizing DataStax Enterprise and Apache Cassandra, leading the company through this critical migration process.

## System Entities and Data Description

The WeAreDoctors system manages various key entities that are essential for its operations. Understanding these entities and their associated data points is crucial for designing an effective NoSQL database solution.

### Customer Info

Each customer within the WeAreDoctors system is uniquely identified by a `CUST_ID` and is registered on a specific `JOINED_DATE`.  The system records the customer's first name and last name, along with their geographical location, specifically the Country and State of residence.  Furthermore, a single customer can hold multiple health service accounts, each identified by a unique `ACCOUNT_ID`, allowing for a flexible relationship between customers and their service engagements.

- CUST_ID (Primary Key)
- FIRST_NAME
- LAST_NAME
- JOINED_DATE
- COUNTRY_ID (Foreign Key to Location Ref)
- STATE_ID (Foreign Key to Location Ref)


### Health Organisation

Within the WeAreDoctors ecosystem, various healthcare service organizations participate to offer a wide spectrum of healthcare services. Each organization is identified by an `ORG_ID` and is categorized by its `ORG_TYPE`, which denotes the nature of services provided, such as general-purpose consultation, specialized weight loss programs, or mental wellness initiatives.  To ensure timely and regionally relevant service delivery, each healthcare organization typically focuses its operations within a specific geographical region of the world, aligning its services with local needs and regulations.

- ORG_ID (Primary Key)
- ORG_TYPE (categorizes services like general consultation, weight loss programs, etc.)
- REGION_ID (Foreign Key to Location Ref)


### Doctor Info

Healthcare practitioners, identified by a `DOC_ID`, are integral to the WeAreDoctors service delivery model.  To provide services through the platform, each practitioner must be recruited by a healthcare organization.  The system captures essential professional details, including the doctor's years of experience (`EXPERIENCE_YEAR`) and age, derived from their birth year (`BIRTH_YEAR`).  Optionally, the system also records professional credentials to validate and showcase practitioner expertise. These credentials include the type of Medical Degree (`CREDENTIAL_M`), such as Medical Doctor (M.D.) or Doctor of Osteopathic Medicine (D.O.), the Medical School graduated from (`CREDENTIAL_S`), for example, Harvard or NUS, and the date the Medical Certification was obtained (`CREDENTIAL_D`).  This detailed information ensures transparency and builds patient trust in the qualifications of the healthcare providers.

- DOC_ID (Primary Key)
- ORG_ID (Foreign Key to Health Organisation)
- FIRST_NAME
- LAST_NAME
- EXPERIENCE_YEAR
- BIRTH_YEAR
- CREDENTIAL_M (type of Medical Degree: M.D., D.O., etc.)
- CREDENTIAL_S (Medical School: Harvard, NUS, etc.)
- CREDENTIAL_D (date Medical Certification was obtained)

### Medical Account

Every customer within the system possesses specific medical accounts (`ACCOUNT_ID`) associated with different healthcare service organizations. These accounts are characterized by their `ACCOUNT_TYPE`, which defines the billing structure.  Examples of account types include 'pay-as-you-visit' models, where customers are charged per consultation, and 'monthly subscription' models, offering fixed-price access to services over a defined period. This variety in account types caters to diverse customer preferences and service utilization patterns.

- ACCOUNT_ID (Primary Key)
- ACCOUNT_TYPE (defines billing structure: pay-as-you-visit, monthly subscription, etc.)
- CUST_ID (Foreign Key to Customer Info)
- ORG_ID (Foreign Key to Health Organisation)

### Medical Records

The system meticulously records medical interactions. When a customer schedules an appointment with a healthcare practitioner (`DOC_ID`), several key details are captured.  These include the appointment date (`TX_DATE`), the patient-stated reason for the appointment, described as `MEDICAL_PURPOSE`, and the associated medical fee (`MEDICAL_AMT`).  These medical records form a comprehensive history of patient interactions and are crucial for both service delivery and administrative purposes.


- MEDICAL_ID (Primary Key)
- ACCOUNT_ID (Foreign Key to Medical Account)
- DOC_ID (Foreign Key to Doctor Info)
- TX_DATE (Primary Key, represents appointment date)
- MEDICAL_PURPOSE (patient-stated reason for appointment)
- MEDICAL_AMT (associated medical fee)

### Organisation Reviews

To ensure service quality and gather customer feedback, the WeAreDoctors platform allows customers to leave reviews for healthcare organizations.  Each review is assigned a unique `REVIEW_ID` and includes textual `REVIEW_COMMENT` providing qualitative feedback.  Customers can also rate the organization on a scale from 0 to 5 stars using `REVIEW_RATING`.  In some instances, reviews may be enriched with multimedia content, such as images, videos, or audio files, collectively captured under `MEDIA`, to provide more detailed and contextual feedback.

- REVIEW_ID (Primary Key)
- REVIEW_DATE
- REVIEW_RATING (scale from 0 to 5 stars)
- REVIEW_COMMENT (qualitative feedback)
- MEDIA (images, videos, or audio files for detailed feedback)
- ORG_ID (Foreign Key to Health Organisation)
- CUST_ID (Foreign Key to Customer Info)


### Location Ref
The system incorporates geographical reference data to manage regional service offerings effectively. This includes information on Regions, such as 'Asia' or 'US', and their associated Countries and the States within each country.  This location data is essential for organizing healthcare organizations by geographical focus and for customer demographic analysis.

- STATE_ID (Primary Key)
- STATE_NAME
- REGION_ID
- REGION_NAME (identifies regions like 'Asia', 'US', etc.)
- COUNTRY_ID
- COUNTRY_NAME


## System Use Cases

The WeAreDoctors system is designed to support various use cases for different categories of users, including customers, doctors/organizations, and system administrators. These use cases highlight the key functionalities and data access patterns required from the system.

### Customer Use Cases

Customers of WeAreDoctors require functionalities that enhance their experience and access to healthcare services. 

Key use cases for customers include the ability to view their top 5 medical appointments based on the highest expenses incurred, enabling them to track their healthcare spending.  

Customers also need to identify popular healthcare practitioners based on the number of medical records associated with each doctor, assisting them in choosing experienced and frequently utilized providers.  

Furthermore, customers should be able to readily access and read the latest offerings of healthcare service organizations without needing to log into the system, facilitating easy discovery of new services and promotions.  

Finally, customers benefit from the ability to monitor their moving average medical expenses over a 3-month period, allowing them to track spending trends and manage their healthcare budget effectively.

### Doctor and Organisation Use Cases

Doctors and healthcare organizations using the WeAreDoctors platform require specific functionalities to manage their services and patient interactions. Doctors and organizations need to be able to review recently completed medical records, allowing them to track patient interactions and follow up on care.

Additionally, they need to view the average ratings provided by customers on a monthly basis, providing valuable feedback on service quality and customer satisfaction trends, enabling them to identify areas for improvement and maintain high service standards.

### System Administrator Use Cases

System administrators for WeAreDoctors require functionalities to manage the overall platform and ensure smooth operations. Administrators need to be able to view the full name of each customer, derived by combining `FIRST_NAME` and `LAST_NAME`, along with a list of all customer accounts, to effectively manage user data and account relationships.
Furthermore, administrators must be able to generate reports on the total number of transactions made by each healthcare organization on a daily basis for the entire year, providing critical insights into system usage, organizational activity levels, and overall platform performance and trends over time.

## Task Overview: Cassandra Solution Design and Implementation

This project aims to design and implement an Apache Cassandra solution, leveraging the DataStax Enterprise platform, to support the WeAreDoctors healthcare system. The Cassandra data modeling methodology, specifically the "Application-Model-Data" approach, will be employed throughout this assignment. This methodology ensures that the data model is driven by the application's requirements and access patterns, leading to an efficient and performant NoSQL database design.  The following tasks outline the key stages of this project, guiding the learner through the end-to-end data modeling and implementation process, ultimately resulting in a robust and scalable Cassandra-based system for WeAreDoctors.

## Task Breakdown

The project is structured into three primary tasks, each focusing on a distinct phase of the Cassandra solution development process.

### Task 1: Conceptual and Application Workflow (10 Marks)

This initial task focuses on the conceptual understanding and application workflow design. 
Learners are required to design the conceptual model based on the provided requirements, effectively translating the business needs into a high-level data representation.
Furthermore, learners must propose suitable application queries and access patterns that directly address the customer and administrator use cases outlined previously. 
This task emphasizes the critical step of understanding the application's needs and how users will interact with the data, forming the foundation for the subsequent data modeling stages.

### Task 2: Logical and Physical Data Modelling (15 Marks)

Building upon the conceptual model from Task 1, this task delves into the logical and physical data modeling aspects.  Learners are expected to propose at least two additional use cases beyond those initially provided, demonstrating their ability to anticipate future requirements and enhance the system's functionality.  
The core of this task involves implementing the most suitable logical and physical data model for Cassandra, utilizing Chebotko Diagramming to visually represent the data model. 
This diagram should comprehensively capture both the initial requirements and the newly proposed use cases, ensuring a robust and scalable data structure within Cassandra.

### Task 3: Solution Implementation and Optimisation (15 Marks)

The final task centers on the practical implementation and optimization of the designed Cassandra solution.  Learners are required to implement a working prototype of the data model on a DataStax Enterprise Virtual Machine (VM), bringing the theoretical design into a functional system.
Crucially, this task also requires learners to propose and implement optimizations aimed at enhancing the overall Cassandra solution's performance, scalability, and fault tolerance.  
These optimizations might include tuning Cassandra configurations, refining data models based on performance testing, and implementing best practices to ensure the system can handle growing data volumes and user traffic efficiently and reliably.


### Task 1: Conceptual and Application Workflow (10 Marks)
Conceptual Diagram

Customer
{ Image }
The customer entity contains joined date and name, which expands to first and last name.

Account
{ Image }
A customer has many medical accounts, and many medical accounts can be owned by a customer.
The account entity contains type, which defines the billing structure.  Examples of account types include 'pay-as-you-visit' models, where customers are charged per consultation, and 'monthly subscription' models, offering fixed-price access to services over a defined period.

Organization
{ Image }
An organization has many accounts, and many accounts are associated with an organization.
The organization entity contains type, which denotes the nature of services provided, such as general-purpose consultation, specialized weight loss programs, or mental wellness initiatives.

Reviews
{ Image }
Many reviews are received by an organization, and an organization receives many reviews.
A customer writes many reviews, and many reviews are written by a customer.
A review contains date, comments, ratings and media.

Records
{ Image }
An account has many medical records, and many medical records are associated with an account.
A record contains appointment date, purpose and amount.

Doctor
{ Image }
An organization affiliates with many doctors, and many doctors are affiliated with an organization.
A doctor contains name, which expands to first and last name, years of experience, birth year, degree, school, and grad date.

State, Country and Region
{ Image }
The location reference model breaks into three hierarchical entities: Region, Country, and State.
Each of these entities has a name.
A state has many customers, and many customers reside in a state.
Many states also belong to a country, and a country contains many states.
Many countries belong to a region, and a region contains many countries.
A region has many organizations, and many organizations operate in a region.


Application Workflow


### Customer Journey
Unknown User is greeted with a landing page
{ Image }
Q1: Top 10 organization offerings (org.type) descending timestamp

Example Schema:
CREATE TABLE organizations_by_update_time (
    time_bucket text, -- 'latest', '2023-Q4', '2023-Q3', etc
    updated_at timestamp,
    org_id uuid,
    org_type text,
    region_id uuid,
    PRIMARY KEY (time_bucket, updated_at, org_id)
) WITH CLUSTERING ORDER BY (updated_at DESC, org_id ASC);

Having the latest bucket will allow all applications to simply call the latest bucket and not have to worry what quarter it is. If most of the read & writes are for latest, it will spread out the partitions evenly compared to storing in current quarter.

However, this means we need an automated system to move data between buckets.

Example Query:
SELECT org_id, org_type, updated_at
FROM organizations_by_update_time
WHERE time_bucket = 'latest'
LIMIT 10;

Customer logs in.
{ Image }
Q2_Extra: Finds user with a specified CUST_ID
Example Schema:
CREATE TABLE customers (
    cust_id uuid,
    first_name text,
    last_name text,
    joined_date timestamp,
    country_id uuid,
    state_id uuid,
    PRIMARY KEY (cust_id)
);

Example Query:
SELECT * FROM customers WHERE cust_id = ?;

Customer would like to identify expensive services for tax purposes and budget planning.
{ Image }
Q3: Top 5 records based on descending order of amount (paginated)
Example Schema:
CREATE TABLE records_by_account_and_amount (
    account_id uuid,
    amount_bucket int, -- Bucketing high amounts for better distribution
    medical_amt decimal,
    medical_id uuid,
    tx_date timestamp,
    medical_purpose text,
    PRIMARY KEY (account_id, amount_bucket, medical_amt, medical_id)
) WITH CLUSTERING ORDER BY (amount_bucket DESC, medical_amt DESC, medical_id ASC);

By adding an amount_bucket (e.g., 1 for >$1000, 2 for $500-$999, 3 for <$500), we distribute records across multiple logical partitions. Now querying the most expensive records only requires checking bucket 1, not scanning all records.

Example Query:
SELECT medical_id, tx_date, medical_purpose, medical_amt
FROM records_by_account_and_amount
WHERE account_id = ? AND amount_bucket = 1 -- Highest amount bucket
LIMIT 5;

Customer is considering to move here and wants to find new care provider with high patient ratings.
{ Image }
Q4: Top 1 doctor descending number of records
Example Schema:
CREATE TABLE doctors_by_record_count (
    count_bucket int, -- For distribution, updated when thresholds crossed
    record_count counter,
    doc_id uuid,
    first_name text,
    last_name text,
    PRIMARY KEY (count_bucket, record_count, doc_id)
) WITH CLUSTERING ORDER BY (record_count DESC, doc_id ASC);

Example Query:
SELECT doc_id, first_name, last_name, record_count
FROM doctors_by_record_count
WHERE count_bucket = 1 -- Highest bucket
LIMIT 1;

Customer is considering changing from pay-per-visit to monthly subscription, and needs to analyze their plan
{ Image }
Q5: Moving Average of medical expenses over 3 months
Example Schema:
CREATE TABLE monthly_expenses_by_account (
    account_id uuid,
    year_month text, -- Format: 'YYYY-MM'
    total_medical_amt decimal,
    PRIMARY KEY (account_id, year_month)
) WITH CLUSTERING ORDER BY (year_month DESC);

Example Query:
SELECT year_month, total_medical_amt 
FROM monthly_expenses_by_account 
WHERE account_id = ? 
LIMIT 3;

Customer is reminded of their latest appointments through SMS.
{ Image }
Q6_Extra: Top 3 records filtering those before date.now and 1 month from date.now descending appointment date
Example Schema:
CREATE TABLE upcoming_appointments_by_account (
    account_id uuid,
    tx_date date,
    tx_time timestamp, -- More precise time within the date
    medical_id uuid,
    doc_id uuid,
    medical_purpose text,
    PRIMARY KEY (account_id, tx_date, tx_time, medical_id)
) WITH CLUSTERING ORDER BY (tx_date ASC, tx_time ASC, medical_id ASC);

Example Query:
SELECT medical_id, doc_id, tx_date, medical_purpose 
FROM upcoming_appointments_by_account 
WHERE account_id = ? 
  AND tx_date >= ? -- Current date
  AND tx_date < ? -- Current date + 1 month
LIMIT 3;


### Doctor Journey
Doctor logs in
{ Image }
Q7_Extra: Finds doctor with a specified DOC_ID
Example Schema:
CREATE TABLE doctors (
    doc_id uuid,
    org_id uuid,
    first_name text,
    last_name text,
    experience_year int,
    birth_year int,
    credential_m text,
    credential_s text,
    credential_d date,
    PRIMARY KEY (doc_id)
);

Example Query:
SELECT * FROM doctors WHERE doc_id = ?;

Doctor wants to review recent patient interactions to ensure proper follow-up care.
{ Image }
Q8: Top 10 records descending timestamp
Example Schema:
CREATE TABLE records_by_doctor_and_date (
    doc_id uuid,
    tx_date date,
    tx_time timestamp,
    medical_id uuid,
    account_id uuid,
    medical_purpose text,
    medical_amt decimal,
    PRIMARY KEY (doc_id, tx_date, tx_time, medical_id)
) WITH CLUSTERING ORDER BY (tx_date DESC, tx_time DESC, medical_id ASC);

Example Query:
SELECT medical_id, account_id, tx_date, medical_purpose, medical_amt 
FROM records_by_doctor_and_date 
WHERE doc_id = ? 
LIMIT 10;


Doctor needs to understand patient satisfaction levels with their affiliated organization on a monthly basis.
{ Image }
Q9: Aggregated average rating of organization
Example Schema:
CREATE TABLE organization_monthly_ratings (
    org_id uuid,
    year_month text,
    avg_rating decimal,
    rating_count counter,
    org_name text,
    org_type text,
    org_address text,
    region_id uuid,
    region_name text,
    PRIMARY KEY (org_id, year_month)
) WITH CLUSTERING ORDER BY (year_month DESC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '12', 
                   'compaction_window_unit': 'MONTHS'};

Example Query:
SELECT org_id, avg_rating FROM organization_avg_ratings WHERE org_id = ?;

Doctor needs to prepare for upcoming appointments and manage their daily schedule.
{ Image }
Q10_Extra: Top 10 upcoming appointments in chronological order
Example Schema:
CREATE TABLE upcoming_appointments_by_doctor (
    doc_id uuid,
    tx_date date,
    tx_time timestamp,
    medical_id uuid,
    account_id uuid,
    medical_purpose text,
    PRIMARY KEY (doc_id, tx_date, tx_time, medical_id)
) WITH CLUSTERING ORDER BY (tx_date ASC, tx_time ASC, medical_id ASC);

Example Query:
SELECT medical_id, account_id, tx_date, medical_purpose 
FROM upcoming_appointments_by_doctor 
WHERE doc_id = ? 
  AND tx_date >= ? -- Current date
LIMIT 10;


### Administration
System admin greeted with overview
{ Image }
Q11: Top 100 full name (FIRST_NAME + LAST_NAME) of customers AND their account_id
Example Schema:
CREATE TABLE accounts_by_created_time (
    time_bucket text, -- For distribution, e.g., 'latest', '2023-10'
    created_at timestamp,
    account_id uuid,
    cust_id uuid,
    first_name text,
    last_name text,
    PRIMARY KEY (time_bucket, created_at, account_id)
) WITH CLUSTERING ORDER BY (created_at DESC, account_id ASC);

Example Query:
SELECT cust_id, first_name, last_name, account_id, created_at 
FROM accounts_by_created_time 
WHERE time_bucket = 'latest' 
LIMIT 100;


System admin needs to monitor platform usage patterns to optimize system resources and performance.
{ Image }
Q12: Daily transaction volume by organization for the current year
Example Schema:
CREATE TABLE daily_transactions_by_org (
    org_id uuid,
    day date,
    transaction_count counter,
    PRIMARY KEY (org_id, day)
) WITH CLUSTERING ORDER BY (day DESC);

Example Query:
SELECT org_id, day, transaction_count 
FROM daily_transactions_by_org 
WHERE org_id = ? 
  AND day >= ? -- First day of current year
LIMIT 50;


System admin needs to identify high-demand regions for targeted infrastructure scaling and support.
{ Image }
Q13_Extra: Top 10 regions by customer population
Example Schema:
CREATE TABLE regions_by_customer_count (
    count_bucket int, -- 1: >1M, 2: 500K-1M, 3: 100K-499K, etc.
    customer_count counter,
    region_id uuid,
    region_name text,
    PRIMARY KEY (count_bucket, customer_count, region_id)
) WITH CLUSTERING ORDER BY (customer_count DESC, region_id ASC);

Example Query:
SELECT region_id, region_name, customer_count 
FROM regions_by_customer_count 
WHERE count_bucket = 1 -- Highest bucket
LIMIT 10;

Organization admin needs to evaluate doctor performance and revenue contribution for bonus allocation.
{ Image }
Q14_Extra: Top 50 revenue-generating doctors
Example Schema:
CREATE TABLE revenue_by_doctor_and_org (
    org_id uuid,
    revenue_bucket int, -- For distribution
    total_revenue decimal,
    doc_id uuid,
    first_name text,
    last_name text,
    PRIMARY KEY (org_id, revenue_bucket, total_revenue, doc_id)
) WITH CLUSTERING ORDER BY (revenue_bucket DESC, total_revenue DESC, doc_id ASC);

Example Query:
SELECT doc_id, first_name, last_name, total_revenue 
FROM revenue_by_doctor_and_org 
WHERE org_id = ? 
  AND revenue_bucket = 1 -- Highest bucket
LIMIT 50;


### Task 2: Logical and Physical Data Modelling (15 Marks)
6 Additional Use Cases

#### Q2_Extra: Customer Authentication and Profile Access
**Query**: SELECT * FROM customers WHERE cust_id = ?;

This fundamental use case is arguably the most frequently executed query in the entire system, serving as the gateway to all customer interactions. Unlike general reporting queries, this operation:

- Directly impacts security and data privacy compliance by enabling proper authentication
- Powers the entire personalized user experience by retrieving profile data
- Serves as the foundation for HIPAA compliance by ensuring proper data access controls
- Represents a critical path operation where performance issues would affect all user interactions
- Must be optimized for sub-millisecond response times to maintain a responsive application

The high-frequency nature of this query makes it an ideal candidate for aggressive caching strategies and denormalized data structures in Cassandra.

#### Q6_Extra: Proactive Appointment Reminder System
**Query**: SELECT medical_id, doc_id, tx_date, medical_purpose FROM medical_records WHERE account_id = ? AND tx_date > CURRENT_DATE AND tx_date < (CURRENT_DATE + INTERVAL '1 month') ORDER BY tx_date DESC LIMIT 3;

This use case directly addresses a critical healthcare industry challenge: missed appointments. Studies show that missed appointments cost the US healthcare system over $150 billion annually. This query:

- Enables automated SMS/email reminders that can reduce no-show rates by up to 30%
- Prevents revenue leakage by ensuring scheduled appointments are kept
- Improves healthcare outcomes through continuity of care
- Creates a competitive advantage through proactive customer communication
- Requires date-range functionality that must be specially optimized in Cassandra

The business impact of this feature extends beyond convenience to directly affecting revenue, patient outcomes, and operational efficiency.

#### Q7_Extra: Doctor Authentication and Portal Access
**Query**: SELECT * FROM doctors WHERE doc_id = ?;

Like customer lookup, this is a foundational security and access control operation that:

- Enables proper authentication for healthcare providers
- Ensures compliance with medical licensing and credential verification requirements
- Serves as the entry point for all doctor-facing features
- Must be highly available even during system maintenance to ensure uninterrupted healthcare delivery
- Is critical for maintaining the integrity of the medical record system

As healthcare professionals rely on immediate system access during patient care, this query represents a zero-downtime requirement for the system architecture.

#### Q10_Extra: Doctor's Daily Schedule Management
**Query**: SELECT medical_id, account_id, tx_date, medical_purpose FROM medical_records WHERE doc_id = ? AND tx_date >= CURRENT_DATE ORDER BY tx_date ASC LIMIT 10;

This use case directly improves clinical workflow efficiency by:

- Enabling doctors to prepare for their upcoming appointments in chronological order
- Reducing patient wait times through better scheduling visibility
- Allowing healthcare providers to prioritize cases based on medical purpose
- Facilitating just-in-time review of patient information before appointments
- Supporting resource allocation for specialty equipment or consultation rooms

The chronological ordering requirement introduces specific data modeling considerations in Cassandra to ensure efficient retrieval without full table scans.

#### Q13_Extra: Geographic Market Analysis
**Query**: SELECT r.region_id, r.region_name, COUNT(c.cust_id) as customer_count FROM location_ref r JOIN countries co ON r.region_id = co.region_id JOIN states s ON co.country_id = s.country_id JOIN customers c ON s.state_id = c.state_id GROUP BY r.region_id, r.region_name ORDER BY customer_count DESC LIMIT 10;

This strategic business intelligence use case enables data-driven expansion planning by:

- Identifying high-density customer regions for targeted marketing and resources
- Guiding doctor recruitment efforts to match regional demand
- Informing infrastructure scaling decisions for regional data centers
- Supporting localization and regional compliance initiatives
- Enabling comparative performance analysis across different markets

The complex join operations required by this query present a particular challenge in Cassandra, requiring denormalized data models or materialized views to achieve acceptable performance.

#### Q14_Extra: Revenue Contribution Analysis
**Query**: SELECT d.doc_id, d.first_name, d.last_name, SUM(mr.medical_amt) as total_revenue FROM doctors d JOIN medical_records mr ON d.doc_id = mr.doc_id WHERE d.org_id = ? GROUP BY d.doc_id, d.first_name, d.last_name ORDER BY total_revenue DESC LIMIT 50;

This financial analytics use case is vital for organizational sustainability by:

- Supporting performance-based compensation models that incentivize quality care
- Identifying top-performing specialties and service types for strategic expansion
- Highlighting doctors who may need additional support or training
- Providing critical inputs for financial forecasting and budgeting
- Enabling organizations to optimize their provider mix for maximum revenue

The aggregation requirements of this query necessitate careful consideration of Cassandra's limitations with GROUP BY operations, potentially requiring pre-aggregation strategies or secondary aggregation systems.

These additional use cases go beyond basic functionality to address critical business needs that directly impact revenue, operational efficiency, user experience, and strategic decision-making. Their implementation will require specialized data modeling techniques in Cassandra to overcome the traditional limitations of NoSQL databases while leveraging their strengths in scalability and availability.

Logical Data Model

I have generated three diagrams. The first is a version where the data models are consolidated, second is a query-optimized approach, and the third is a query-optimized approach with UDTs.
While the diagrams include data-types, which is not usually included in logical models, it will only be discussed in the Physical Data model section.

## Table Consolidation Approach (First Diagram)
{ Image }
### Key Characteristics:
Combines related functionalities into fewer tables

Uses generic columns (entity_id, owner_id) to support multiple access patterns

Reduces total table count from ~15 to 11

### Advantages:
Fewer tables to maintain and monitor
Reduced data duplication -> more efficient storage
Simpler data synchronization

### Disadvantages:
More complex application logic, because code must interpret generic columns correctly
Potentially reduced query performance, because tables may contain more data than needed for specific queries
Less intuitive, because generic column names require documentation to understand
Harder to optimize, because performance tuning affects multiple query patterns simultaneously

## Query-Optimized Approach (Second Diagram)
{Image}
### Key Characteristics:
- Creates dedicated tables for each access pattern
- column names are more specific
- Results in more total tables (~15)

### Advantages:
Optimized for query performance, since each table is precisely designed for its specific query
Less complex application logic.
More intuitive
True to cassandra's query-first design principles
Easier to optimize.

### Disadvantages:
More tables to maintain and monitor
More data duplication
Complex data duplication

## Query-Optimized Approach with UDTs (Third Diagram)
{Image}
UDTs offer organizational benefits, but they must be controlled such that it doesn't degrade query performance.

The additional UDTs are minimal, including credentials and name, only to serve to organize some of the fields

## Evaluation

The query-optimized approach aligns more closely with Cassandra's fundamental design principles, where data models are driven by specific query patterns rather than normalized data relationships. This is especially important in large-scale, high-throughput systems where query performance is critical.

However, the consolidated approach offers practical advantages for teams with limited resources or for systems with moderate performance requirements. It represents a reasonable compromise between the relational mindset (fewer tables, less duplication) and the Cassandra mindset (query-driven design).

Most production Cassandra implementations ultimately fall somewhere between these two extremes, with critical queries getting dedicated tables while related, less-frequent access patterns might share tables.

For an academic assignment focused on demonstrating understanding of Cassandra data modeling principles, the query-optimized approach more clearly demonstrates mastery of the subject matter.

Physical Data Model

If you would like to look at the CREATE TABLE statements, please look into create_table.sh. 

If you would also like to look at the COPY statements, please look into load_data.sh.

## Physical Implementation Considerations

### 1. Keyspace Definition

CREATE KEYSPACE wearedoctors 
WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'datacenter1': 2,
    'datacenter2': 1
}
AND durable_writes = true;

This keyspace definition:
Uses NetworkTopologyStrategy for multi-datacenter deployment
Configures 2 replicas in primary datacenter and 1 in secondary
Ensures durable writes for critical healthcare data

### 2. Compaction Strategies

Notice that tables follow different compaction strategies based on access patterns:

- TimeWindowCompactionStrategy: Used for time-series data (appointments, records, expenses) to improve read performance on recent data and efficiently compact older data
- SizeTieredCompactionStrategy: Used for more static reference data (customers, doctors)

### 4. Counter Tables Considerations

For counter tables like `doctors_by_record_count`:
- Updates must be made with INCREMENT/DECREMENT operations
- Cannot be included in batches with non-counter updates
- Require special handling for consistency

UPDATE doctors_by_record_count 
SET record_count = record_count + 1 
WHERE count_bucket = ? AND doc_id = ?;

### 7. Bucketing Implementation Details

The bucketing strategy for columns like `amount_bucket` would be implemented in application code:

// Example Java logic for determining amount_bucket
int determineAmountBucket(BigDecimal amount) {
    if (amount.compareTo(new BigDecimal("1000")) >= 0) return 1;
    if (amount.compareTo(new BigDecimal("500")) >= 0) return 2;
    return 3;
}

# Evaluation of Cassandra Data Loading Methods

When migrating large datasets to Apache Cassandra or DataStax Enterprise (DSE), selecting the appropriate loading method is critical for efficiency and performance.

## COPY Command

The COPY command is a native CQL (Cassandra Query Language) utility that functions similarly to PostgreSQL's COPY command, allowing direct data import from CSV files into Cassandra tables.

### Performance Characteristics
The COPY command operates as a single-threaded process through the cqlsh client, which means it follows the normal write path through the coordinator node. This creates several performance limitations:

- Data must be parsed and validated by the client
- Each row generates network traffic between the client and the cluster
- The command uses the standard insert process, including memtables and commit logs
- Throughput is typically limited to 50,000-100,000 rows per minute depending on row size

### Data Consistency and Error Handling
COPY provides basic error handling with options to:
- Skip bad records (using the SKIPROWS parameter)
- Set a maximum error threshold before aborting
- Record failed rows in a separate file for later review

However, it lacks sophisticated error recovery mechanisms, making it less suitable for production-critical loads.

### Scalability Limitations
As a client-side, single-threaded operation, COPY does not scale well for large datasets. The process becomes increasingly inefficient as data volume grows, with several bottlenecks:

- Client memory constraints
- Network bandwidth between client and cluster
- Single-threaded processing

### Ideal Use Cases
COPY is best suited for:
- Development and testing environments
- Small to medium datasets (up to a few GB)
- Simple data structures with minimal transformation needs
- Ad-hoc data loading tasks
- Scenarios where operational simplicity outweighs performance requirements

## SSTable Loader

The SSTable loader is a specialized tool that bypasses the normal write path by loading pre-generated SSTable files directly into Cassandra's storage layer.

### Performance Characteristics
SSTable loader offers exceptional performance advantages:
- Bypasses the entire write path (memtables, commit logs)
- Directly writes to the storage layer, reducing cluster CPU and memory overhead
- Achieves throughput rates 5-10 times faster than COPY command
- Minimizes impact on cluster performance during loading
- Supports streaming to multiple nodes simultaneously

In benchmark tests, SSTable loader consistently handles millions of rows per minute, making it suitable for large-scale data migrations.

### Data Consistency and Error Handling
SSTable loader operates at a lower level in the architecture, which affects data handling:
- Requires pre-validated data as there's minimal validation during loading
- Limited error recovery capabilities
- Requires manual intervention if loading fails
- Strong understanding of Cassandra's data distribution model is necessary

### Scalability Advantages
This method scales exceptionally well for very large datasets because:
- It distributes load directly to appropriate nodes based on token ranges
- It avoids overloading the coordinator node
- It can process multiple SSTables in parallel
- It works efficiently with Cassandra's native file format

### Ideal Use Cases
SSTable loader is optimal for:
- Production migrations of very large datasets (multiple GB to TB)
- Restoring from backups
- Situations where minimizing cluster impact during loading is critical
- Cases where you can pre-generate valid SSTables
- Environments with significant hardware resources dedicated to the loading process

## DSE Bulk Loader

The DSE Bulk Loader is DataStax Enterprise's premium data loading utility, offering advanced features beyond the open-source tools.

### Performance Characteristics
DSE Bulk Loader combines high performance with flexibility:
- Multi-threaded parallel loading capabilities
- Configurable batch sizes and concurrency levels
- Sophisticated connection pooling to optimize cluster communication
- Tunable consistency levels per loading job
- Support for various input formats (CSV, JSON, etc.)
- Performance typically falls between COPY (slower) and SSTable loader (faster)

The tool can be configured to maximize throughput based on available resources, typically achieving hundreds of thousands of rows per minute.

### Data Consistency and Error Handling
This enterprise tool provides comprehensive error management features:
- Detailed logging of all operations
- Configurable error thresholds and handling strategies
- Transaction-like semantics with batch operations
- Ability to resume failed loads from the point of failure
- Validation and transformation capabilities during loading

### Scalability and Integration
DSE Bulk Loader excels in enterprise environments through:
- Seamless integration with other DSE tools
- Support for DSE security features (authentication, encryption)
- Ability to scale across multiple loader processes
- Monitoring integration for operational visibility
- Support for complex ETL workflows

### Ideal Use Cases
DSE Bulk Loader is best for:
- Enterprise production environments with complex requirements
- Regular scheduled data loads as part of ETL processes
- Scenarios requiring sophisticated error handling and reporting
- Data loads requiring transformation during import
- Organizations already invested in the DataStax Enterprise ecosystem

## Comparative Analysis and Recommendations

When selecting a loading method for a healthcare system migration like WeAreDoctors, several factors should be considered:

### For Development and Testing
The COPY command is recommended due to its simplicity and ease of use. It requires minimal setup and works well for iterative development when data volumes are smaller.

### For Initial Production Migration
The SSTable loader offers the best performance for one-time large migrations. For the WeAreDoctors case, this would be ideal for the initial data migration, particularly for historical medical records which may represent the largest portion of data.

### For Ongoing Operational Data Loads
The DSE Bulk Loader provides the best balance of performance, reliability, and manageability for regular operational data loads. For WeAreDoctors, this would be appropriate for regular integration of new patient data, medical records, or system updates.

## Conclusion

For the WeAreDoctors healthcare system migration, a hybrid approach is recommended:

1. Use the COPY command during development phases and for smaller reference tables (e.g., Location Ref, Organization types)

2. Employ the SSTable loader for the initial bulk migration of historical data, particularly for large tables like Medical Records and Customer Info

3. Implement the DSE Bulk Loader for ongoing operational data loads and for tables requiring complex transformations or validation during loading

However, for this assignment, the COPY command will be used.


### Task 3: Solution Implementation and Optimisation (15 Marks)
Solution Implementation
Enable shared folders on VM, and set the path to the vm_shared directory seen at this directory level.

Run the scripts in sequence:
1. create_table.sh
2. load_data.sh
3. select.sh

If there's a mistake run the following:
1. reset.sh

The scripts will automatically prompt to run the next script.

Make sure that data/ is inside vm_shared/. The csv data is synthetically generated with Generative AI, and has been designed to comply with the table defined Cassandra data types.

Optimisations made
## 1. Data Modeling Optimizations

### 1.1 User-Defined Types (UDTs)

The implementation utilizes User-Defined Types to maintain data consistency across multiple tables:

CREATE TYPE IF NOT EXISTS person_name (first_name text, last_name text);
CREATE TYPE IF NOT EXISTS doctor_credential (degree_type text, institution text, certification_date date);

These UDTs are pivotal because they:
- Ensure consistent representation of complex data structures across multiple tables
- Simplify schema management by centralizing the definition of common data structures
- Enhance code readability and maintenance by providing semantic meaning to complex fields
- Support healthcare compliance requirements by standardizing how personally identifiable information is stored

### 1.2 Denormalized Data Model

The implementation extensively employs denormalization, which is essential in Cassandra's distributed architecture:

- Doctor information appears in at least five different tables (`doctors`, `doctors_by_popularity`, `records_by_doctor_and_date`, etc.)
- Organization details are duplicated across multiple tables
- Customer information is replicated where needed for direct access

This approach is pivotal because:
- It eliminates expensive joins that are not supported in Cassandra
- Supports query-first design methodology where data duplication is acceptable to optimize read performance
- Enables single-partition queries that are significantly more efficient in distributed environments
- Addresses healthcare systems' need for rapid data access during critical care scenarios

## 2. Physical Storage Optimizations

### 2.1 Strategic Compaction Strategies

The implementation carefully selects appropriate compaction strategies for different data access patterns:

#### TimeWindowCompactionStrategy
Applied to time-series data with clear retention needs:
WITH compaction = {'class': 'TimeWindowCompactionStrategy', 
                  'compaction_window_size': '1', 
                  'compaction_window_unit': 'DAYS'}

Used in tables like `organizations_by_update_time`, `monthly_expenses_by_account`, and `upcoming_appointments_by_account`.

#### SizeTieredCompactionStrategy
Applied to more static data or where time is not the primary organization factor:
WITH compaction = {'class': 'SizeTieredCompactionStrategy'}

Used in tables like `customers`, `doctors`, and `organization_ratings`.

This differentiation is pivotal because:
- TimeWindowCompactionStrategy optimizes storage of time-series data by compacting files with data from the same time window
- It significantly improves read performance for recent data, which is critical for appointment scheduling and current patient care
- SizeTieredCompactionStrategy works better for relatively static data like doctor credentials or customer profiles
- The right compaction strategy can dramatically reduce disk I/O and improve query latency

### 2.2 Strategic Clustering Order Specifications

Every table includes carefully designed clustering orders:
WITH CLUSTERING ORDER BY (updated_at DESC, org_id ASC)
WITH CLUSTERING ORDER BY (amount_bucket DESC, medical_amt DESC, medical_id ASC)
WITH CLUSTERING ORDER BY (tx_time ASC, medical_id ASC)

These specifications are pivotal because:
- They determine the physical ordering of data on disk within each partition
- Enable efficient range queries without requiring in-memory sorting
- Support natural access patterns (most recent items first, highest amounts first, upcoming appointments in chronological order)
- Critical for healthcare applications where time-ordered medical records and appointment schedules are essential

## 3. Query Pattern Optimizations

### 3.1 Bucketing Strategies

The implementation employs multiple bucketing approaches to optimize data distribution:

- Time buckets: `time_bucket` in `organizations_by_update_time` and `accounts_by_created_time`
- Amount buckets: `amount_bucket` in `records_by_account_and_amount`
- Count buckets: `count_bucket` in `doctors_by_popularity` and `regions_by_popularity`
- Revenue buckets: `revenue_bucket` in `revenue_by_doctor_and_org`

This bucketing approach is pivotal because:
- It prevents "hot partitions" that can create performance bottlenecks in distributed systems
- Spreads writes and reads evenly across the cluster, improving overall system throughput
- Allows for efficient range-based queries within manageable partition sizes
- Enables scaling to handle the vast amounts of medical record data generated in healthcare settings

### 3.2 Counter Tables

The implementation uses dedicated counter tables for efficient statistics tracking:
CREATE TABLE IF NOT EXISTS doctor_record_counts (
    count_bucket int,
    doc_id uuid,
    record_count counter,
    PRIMARY KEY (count_bucket, doc_id)
)

These specialized tables are pivotal because:
- They leverage Cassandra's native counter type for atomic increment/decrement operations
- Enable efficient high-concurrency updates for metrics tracking
- Separate write-heavy counter operations from read-heavy data access
- Support accurate analytics for doctor popularity and organization activity metrics that inform healthcare decision-making

### 3.3 Pre-aggregation Tables

Several tables store pre-calculated aggregates rather than computing them at query time:
- `monthly_expenses_by_account`: Pre-calculates monthly financial totals
- `organization_ratings`: Stores pre-calculated average ratings
- `daily_org_transactions`: Maintains daily revenue totals

This approach is pivotal because:
- Cassandra lacks built-in aggregation functions, making on-the-fly calculations expensive
- Pre-aggregation dramatically improves read performance for commonly accessed metrics
- Enables efficient time-series analysis and reporting without complex client-side processing
- Supports healthcare billing and financial reporting requirements with minimal query latency

## 4. Data Access Optimization

### 4.1 Table-Per-Query Pattern

The implementation creates dedicated tables for specific query patterns:
- `upcoming_appointments_by_account` for patient appointment viewing
- `upcoming_appointments_by_doctor` for doctor schedule management
- `records_by_account_and_amount` for expense tracking
- `records_by_doctor_and_date` for medical history

This approach is pivotal because:
- It aligns perfectly with Cassandra's query-first design philosophy
- Eliminates the need for secondary indexes which perform poorly at scale
- Each table is optimized for a specific access pattern with appropriate partition and clustering keys
- Supports different stakeholder needs (patients, doctors, administrators) with optimized data structures

### 4.2 Optimized Loading Procedures

The data loading process includes special handling for counter tables:
awk -F, 'NR>1 {print "UPDATE '$KEYSPACE_NAME'.doctor_record_counts SET record_count = record_count + "$3" WHERE count_bucket = "$1" AND doc_id = "$2";"}' "$CSV_DIR/doctor_record_counts.csv" >/tmp/doctor_record_counts.cql

This specialized approach is pivotal because:
- Counter columns require UPDATE statements rather than direct insertion
- Properly initializes counter values while maintaining data integrity
- Supports efficient data loading even for specialized Cassandra data types
- Enables reliable system initialization with consistent performance characteristics

## 5. Replication Strategy

The implementation configures an appropriate replication strategy:
CREATE KEYSPACE IF NOT EXISTS $KEYSPACE_NAME WITH replication = 
{'class': 'SimpleStrategy', 'replication_factor': 3};

This configuration is pivotal because:
- It ensures data availability even if individual nodes fail
- Provides redundancy appropriate for healthcare data where availability is critical
- Balances availability needs with storage efficiency
- Supports healthcare compliance requirements for data redundancy and disaster recovery
