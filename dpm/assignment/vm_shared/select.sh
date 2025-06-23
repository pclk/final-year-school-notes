#!/bin/bash

# Set variables
KEYSPACE_NAME="wearedoctors"

# Set colors for output
GREEN='\033[0;32m'
BLUE='\033[1;36m'
YELLOW='\033[0;33m' # Added for SQL statement highlighting
NC='\033[0m'        # No Color

# Function to run a CQL query with proper error handling and display the SQL
run_query() {
  local query="$1"
  local description="$2"
  local importance="$3"

  echo -e "${BLUE}$description:${NC}"
  echo -e "${BLUE}$importance${NC}"
  echo -e "${YELLOW}SQL:${NC}"
  echo -e "${YELLOW}-----${NC}"
  # Print the SQL with proper indentation preserved
  echo -e "${YELLOW}USE $KEYSPACE_NAME;${NC}"
  echo -e "${YELLOW}$query${NC}"
  echo -e "${YELLOW}-----${NC}"

  if ! output=$(cqlsh -e "USE $KEYSPACE_NAME; $query" 2>&1); then
    echo "Error executing query: $query"
    echo "$output"
    return 1
  fi
  echo "$output"
  echo -e "-----------------------------------------\n"
}

echo -e "${GREEN}Testing SELECT statements on each table...${NC}"

# Hardcoded sample filter_by variables:
TIME_BUCKET='2024-01'
CUST_ID='1a2b3c4d-e5f6-7890-1234-567890abcdef'
ACCOUNT_ID='a1b2c3d4-e5f6-7890-1234-567890abcdef'
COUNT_BUCKET='10'
DOC_ID='d1d2d3d4-e5f6-7890-1234-567890abcdef'
ORG_ID='f1f2f3f4-e5f6-7890-1234-567890abcdef'
ACC_TIME_BUCKET='2024-01'
REGION_COUNT_BUCKET='10'

# 1. organizations_by_update_time
echo -e "\n${GREEN}1. Testing organizations_by_update_time table...${NC}"
run_query "SELECT org_name, org_type, org_address, region_name, updated_at 
FROM organizations_by_update_time 
LIMIT 10;" "Get the most recently updated healthcare organizations" "This query supports the critical customer use case where 'customers should be able to readily access and read the latest offerings of healthcare service organizations without needing to log into the system'. By providing updated organization information, the system helps prospective patients discover relevant healthcare providers before registration, enhancing service accessibility. The time-bucketed design optimizes Cassandra performance for frequently accessed recent data."

run_query "SELECT org_name, org_address, updated_at, region_name, org_type 
FROM organizations_by_update_time 
WHERE time_bucket = '$TIME_BUCKET' 
LIMIT 10;" "Organizations in time bucket $TIME_BUCKET" "This query implements the time bucketing strategy described in the schema, which distributes data evenly across partitions to prevent hotspots in Cassandra. By filtering by time bucket, the system can efficiently retrieve organizations updated within specific time periods, allowing customers to browse relevant healthcare options chronologically without scanning the entire dataset."

# 2. customers
echo -e "\n${GREEN}2. Testing customers table...${NC}"
run_query "SELECT cust_id, cust_name, joined_date, country_name, state_name, region_name 
FROM customers 
LIMIT 5;" "Retrieve sample customer details" "This query supports the administrator use case where 'Administrators need to be able to view the full name of each customer'. Customer profile access is fundamental for user management, identity verification, and compliance with healthcare regulations that require proper patient identification. This table uses a SizeTieredCompactionStrategy optimized for relatively static user profile data."

run_query "SELECT cust_name, joined_date, country_name, state_name, region_name 
FROM customers 
WHERE cust_id = $CUST_ID;" "Details for specific customer" "This query implements the Customer Authentication use case, which is 'arguably the most frequently executed query in the entire system, serving as the gateway to all customer interactions'. Direct lookups by primary key are optimized in Cassandra, enabling sub-millisecond response times necessary for a responsive application. This query 'powers the entire personalized user experience' and serves as 'the foundation for HIPAA compliance by ensuring proper data access controls'."

# 3. records_by_account_and_amount
echo -e "\n${GREEN}3. Testing records_by_account_and_amount table...${NC}"
run_query "SELECT account_id, amount_bucket, medical_amt, medical_purpose, tx_date, doc_name 
FROM records_by_account_and_amount 
LIMIT 5;" "Find sample medical transactions" "This query enables access to medical transaction records, which form the core of the healthcare system's data. It provides visibility into healthcare services rendered, their costs, and associated healthcare providers. The table's structure with amount bucketing allows for efficient data distribution while maintaining the ability to quickly retrieve records, supporting both operational and analytical needs."

run_query "SELECT amount_bucket, medical_amt, medical_purpose, tx_date, doc_name, org_name 
FROM records_by_account_and_amount 
WHERE account_id = $ACCOUNT_ID 
LIMIT 5;" "Medical transactions for specific account" "This query directly supports the customer use case to 'view their top 5 medical appointments based on the highest expenses incurred'. It enables patients to track healthcare spending for tax purposes and budget planning. The amount bucketing strategy  optimizes query performance by distributing records across multiple logical partitions, ensuring that retrieving the most expensive records doesn't require scanning all transactions."

# 4. doctors_by_popularity and doctor_record_counts
echo -e "\n${GREEN}4a. Testing doctors_by_popularity table...${NC}"
run_query "SELECT count_bucket, doc_id, doc_name, org_name, org_type, experience_year 
FROM doctors_by_popularity 
LIMIT 5;" "View popular doctors" "This query supports the customer need to 'identify popular healthcare practitioners', which helps patients make informed decisions when selecting providers. Doctor popularity, as measured by patient volume, can be an indicator of quality and experience. The count_bucket partitioning strategy optimizes the distribution of doctors across the cluster while maintaining the ability to quickly identify the most popular practitioners."

run_query "SELECT doc_id, doc_name, org_name, org_type, experience_year 
FROM doctors_by_popularity 
WHERE count_bucket = $COUNT_BUCKET 
LIMIT 10;" "Doctors in popularity bucket $COUNT_BUCKET" "This query enables patients to find highly experienced doctors within specific volume categories. As noted in the case, customers need to find 'experienced and frequently utilized providers'. By organizing doctors into count buckets, the system can efficiently respond to queries for doctors with similar levels of patient interaction, supporting informed healthcare provider selection without requiring sorting across the entire dataset."

echo -e "\n${GREEN}4b. Testing doctor_record_counts table...${NC}"
run_query "SELECT count_bucket, doc_id, record_count 
FROM doctor_record_counts 
LIMIT 5;" "View doctor record counts" "This query provides visibility into the patient volume handled by different doctors, supporting both operational monitoring and customer decision-making. Counter tables in Cassandra offer efficient increment/decrement operations, making them ideal for tracking doctor popularity metrics that require frequent updates as new medical records are created."

# 5. monthly_expenses_by_account
echo -e "\n${GREEN}5. Testing monthly_expenses_by_account table...${NC}"
run_query "SELECT account_id, year_month, total_medical_amt, transaction_count 
FROM monthly_expenses_by_account 
LIMIT 5;" "View sample monthly expenses" "This query provides access to pre-aggregated monthly expense data, which is essential for efficient time-series analysis in Cassandra. By maintaining monthly rollups, the system avoids expensive aggregation operations at query time, allowing for quick access to historical spending patterns while reducing computational overhead."

run_query "SELECT year_month, total_medical_amt, transaction_count 
FROM monthly_expenses_by_account 
WHERE account_id = $ACCOUNT_ID 
LIMIT 6;" "Monthly expenses for specific account" "This query implements the customer use case to 'monitor their moving average medical expenses over a 3-month period', enabling financial planning and budget management. As noted in line 274, customers use this data when 'considering changing from pay-per-visit to monthly subscription' payment models. The TimeWindowCompactionStrategy optimization ensures efficient storage and retrieval of time-series financial data."

# 6. upcoming_appointments_by_account
echo -e "\n${GREEN}6. Testing upcoming_appointments_by_account table...${NC}"
run_query "SELECT account_id, tx_time, doc_name, medical_purpose, org_name 
FROM upcoming_appointments_by_account 
LIMIT 5;" "View sample upcoming appointments" "This query provides access to appointment scheduling data, which is crucial for operational efficiency in healthcare delivery. The table design with tx_date and tx_time as clustering keys optimizes chronological access patterns, ensuring that appointment listings appear in the correct order without requiring client-side sorting."

run_query "SELECT tx_time, doc_name, medical_purpose, org_name 
FROM upcoming_appointments_by_account 
WHERE account_id = $ACCOUNT_ID 
LIMIT 5;" "Upcoming appointments for specific account" "This query implements the Proactive Appointment Reminder System use case, which 'directly addresses a critical healthcare industry challenge: missed appointments' that 'cost the US healthcare system over 150 billion annually'. By enabling automated reminders, this functionality 'can reduce no-show rates by up to 30%', 'prevents revenue leakage', and 'improves healthcare outcomes through continuity of care'."

# 7. doctors
echo -e "\n${GREEN}7. Testing doctors table...${NC}"
run_query "SELECT doc_id, doc_name, org_name, org_type, experience_year, credential 
FROM doctors 
LIMIT 5;" "View sample doctor details" "This query provides access to the comprehensive doctor profile information that the system captures, including 'professional credentials to validate and showcase practitioner expertise'. Maintaining detailed practitioner information ensures transparency and builds patient trust, which is essential for a healthcare platform's credibility and adoption."

run_query "SELECT doc_name, org_name, org_type, experience_year, credential, birth_year 
FROM doctors 
WHERE doc_id = $DOC_ID;" "Details for specific doctor" "This query implements the Doctor Authentication use case, a 'foundational security and access control operation' that 'enables proper authentication for healthcare providers' and 'ensures compliance with medical licensing and credential verification requirements'. This represents a 'zero-downtime requirement for the system architecture' as 'healthcare professionals rely on immediate system access during patient care'."

# 8. records_by_doctor_and_date
echo -e "\n${GREEN}8. Testing records_by_doctor_and_date table...${NC}"
run_query "SELECT doc_id, tx_time, medical_id, cust_name, medical_purpose, medical_amt 
FROM records_by_doctor_and_date 
LIMIT 5;" "View sample medical records by doctor" "This query provides access to healthcare interaction records organized by provider, supporting clinical workflows and administrative oversight. The combination of doc_id as the partition key with temporal clustering ensures efficient retrieval of a doctor's patient interactions while maintaining chronological order, which is essential for medical record management."

run_query "SELECT tx_time, medical_id, cust_name, medical_purpose, medical_amt 
FROM records_by_doctor_and_date 
WHERE doc_id = $DOC_ID 
LIMIT 10;" "Medical records for specific doctor" "This query supports the doctor use case to 'review recently completed medical records', which is 'essential for continuity of care, allowing doctors to track patient interactions and follow up on care'. The TimeWindowCompactionStrategy optimization ensures efficient storage and retrieval of recent medical records, which are accessed more frequently than historical data."

# 9. organization_ratings
echo -e "\n${GREEN}9. Testing organization_ratings table...${NC}"
run_query "SELECT org_id, org_name, org_type, avg_rating, region_name 
FROM organization_ratings 
LIMIT 5;" "View sample organization ratings" "This query provides access to aggregated customer satisfaction metrics, which are essential for service quality monitoring and improvement. The organization ratings table implements the platform's review system where 'customers can rate the organization on a scale from 0 to 5 stars', offering transparency and accountability in healthcare service delivery."

run_query "SELECT org_name, org_type, avg_rating, org_address, region_name 
FROM organization_ratings 
WHERE org_id = $ORG_ID;" "Rating for specific organization" "This query implements the doctor and organization use case to 'view the average ratings provided by customers', providing 'valuable feedback on service quality and customer satisfaction trends, enabling them to identify areas for improvement and maintain high service standards'. These satisfaction metrics are critical for healthcare organizations to evaluate their market perception and service quality."

# 10. upcoming_appointments_by_doctor
echo -e "\n${GREEN}10. Testing upcoming_appointments_by_doctor table...${NC}"
run_query "SELECT doc_id, tx_time, cust_name, medical_purpose 
FROM upcoming_appointments_by_doctor 
LIMIT 5;" "View sample upcoming appointments by doctor" "This query provides access to scheduled appointments organized by healthcare provider, which is essential for clinical workflow management. The table design optimizes for chronological access patterns, ensuring that doctors can easily view their upcoming appointments in the order they will occur without requiring client-side sorting."

run_query "SELECT tx_time, medical_id, cust_name, medical_purpose 
FROM upcoming_appointments_by_doctor 
WHERE doc_id = $DOC_ID 
LIMIT 10;" "Upcoming appointments for specific doctor" "This query supports the Doctor's Daily Schedule Management use case, which 'directly improves clinical workflow efficiency' by 'enabling doctors to prepare for their upcoming appointments in chronological order'. This functionality helps in 'reducing patient wait times', 'allowing healthcare providers to prioritize cases based on medical purpose', and 'facilitating just-in-time review of patient information before appointments'."

# 11. accounts_by_created_time
echo -e "\n${GREEN}11. Testing accounts_by_created_time table...${NC}"
run_query "SELECT time_bucket, created_at, account_id, cust_name, region_name 
FROM accounts_by_created_time 
LIMIT 5;" "View sample recently created accounts" "This query provides chronological visibility into account creation, which is essential for monitoring system growth and user acquisition patterns. The time-bucketed design optimizes Cassandra performance for time-series data access, ensuring efficient retrieval of recent accounts without scanning the entire dataset."

run_query "SELECT created_at, account_id, cust_name, country_name, region_name 
FROM accounts_by_created_time 
WHERE time_bucket = '$ACC_TIME_BUCKET' 
LIMIT 10;" "Recently created accounts in time bucket $ACC_TIME_BUCKET" "This query implements the administrator use case to 'view the full name of each customer... along with a list of all customer accounts'. By organizing accounts chronologically with time bucketing, administrators can efficiently monitor recent system activity and user onboarding, providing critical insights for operational oversight and capacity planning."

# 12. daily_org_transactions and daily_org_transaction_counts
echo -e "\n${GREEN}12a. Testing daily_org_transactions table...${NC}"
run_query "SELECT org_id, day, total_revenue, org_name, org_type 
FROM daily_org_transactions 
WHERE org_id = $ORG_ID 
LIMIT 10;" "Daily transactions for specific organization" "This query supports the administrator use case to 'generate reports on the total number of transactions made by each healthcare organization on a daily basis for the entire year'. By maintaining pre-aggregated daily revenue figures, the system enables efficient financial reporting without requiring expensive aggregation operations at query time."

echo -e "\n${GREEN}12b. Testing daily_org_transaction_counts table...${NC}"
run_query "SELECT org_id, day, transaction_count 
FROM daily_org_transaction_counts 
LIMIT 5;" "View sample transaction counts" "This query provides access to daily operational metrics, offering insights into platform usage patterns and organizational activity levels. The counter table design optimizes for frequent updates while maintaining efficient read access, which is essential for real-time monitoring of system utilization."

run_query "SELECT day, transaction_count 
FROM daily_org_transaction_counts 
WHERE org_id = $ORG_ID 
LIMIT 10;" "Daily transaction counts for specific organization" "This query supports administrators in monitoring 'platform usage, organizational activity levels, and overall platform performance and trends over time'. The TimeWindowCompactionStrategy optimization ensures efficient storage and retrieval of recent transaction data, which is accessed more frequently than historical records, enabling effective operational monitoring."

# 13. regions_by_popularity and region_customer_counts
echo -e "\n${GREEN}13a. Testing regions_by_popularity table...${NC}"
run_query "SELECT count_bucket, region_id, region_name, country_name, total_revenue 
FROM regions_by_popularity 
LIMIT 5;" "View sample popular regions" "This query provides insights into geographical distribution of customers and revenue, supporting strategic business decision-making. The count-bucketed design optimizes the distribution of regions across the cluster while maintaining the ability to quickly identify the most populated areas, enabling efficient market analysis."

run_query "SELECT count_bucket, region_name, country_name, total_revenue 
FROM regions_by_popularity 
WHERE count_bucket = $COUNT_BUCKET
LIMIT 10;" "Details for specific region" "This query implements the Geographic Market Analysis use case, which 'enables data-driven expansion planning' by 'identifying high-density customer regions for targeted marketing and resources' and 'guiding doctor recruitment efforts to match regional demand'. This strategic information is essential for 'infrastructure scaling decisions' and 'supporting localization and regional compliance initiatives'."

echo -e "\n${GREEN}13b. Testing region_customer_counts table...${NC}"
run_query "SELECT count_bucket, region_id, customer_count 
FROM region_customer_counts 
LIMIT 5;" "View sample region customer counts" "This query provides access to customer population metrics by region, which is essential for geographic market analysis and strategic planning. The counter table design optimizes for frequent updates while maintaining efficient read access, supporting real-time monitoring of regional customer distribution."

run_query "SELECT region_id, customer_count 
FROM region_customer_counts 
WHERE count_bucket = $REGION_COUNT_BUCKET 
LIMIT 10;" "Customer counts in region bucket $REGION_COUNT_BUCKET" "This query supports the identification of 'high-density customer regions for targeted marketing and resources'. By maintaining separate counter tables for customer counts, the system follows Cassandra best practices for efficiently tracking frequently updated metrics while allowing quick access to regions with similar population levels."

# 14. revenue_by_doctor_and_org
echo -e "\n${GREEN}14. Testing revenue_by_doctor_and_org table...${NC}"
run_query "SELECT org_id, revenue_bucket, doc_name, total_revenue, transaction_count 
FROM revenue_by_doctor_and_org 
LIMIT 5;" "View sample doctor revenue data" "This query provides access to doctor performance metrics in terms of revenue generation, which is essential for financial analysis and incentive management. The organization and bucket-based partitioning ensures efficient data distribution while enabling quick identification of top performers within each organization."

run_query "SELECT revenue_bucket, doc_name, total_revenue, transaction_count 
FROM revenue_by_doctor_and_org 
WHERE org_id = $ORG_ID 
LIMIT 10;" "Revenue data for doctors in specific organization" "This query implements the Revenue Contribution Analysis use case, which is 'vital for organizational sustainability' by 'supporting performance-based compensation models that incentivize quality care' and 'identifying top-performing specialties and service types for strategic expansion'. This financial intelligence is crucial for 'providing critical inputs for financial forecasting and budgeting'."

echo -e "\n${GREEN}All SELECT tests completed!${NC}"
