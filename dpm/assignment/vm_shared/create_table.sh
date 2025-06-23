#!/bin/bash

# Set variables
KEYSPACE_NAME="wearedoctors"

# Create keyspace and tables
echo "Creating keyspace and tables..."

# Create keyspace
cqlsh -e "CREATE KEYSPACE IF NOT EXISTS $KEYSPACE_NAME WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};"

# Use the keyspace for all subsequent commands
echo "Creating UDTs..."

# Create UDTs
cqlsh -e "USE $KEYSPACE_NAME; CREATE TYPE IF NOT EXISTS person_name (first_name text, last_name text);"
cqlsh -e "USE $KEYSPACE_NAME; CREATE TYPE IF NOT EXISTS doctor_credential (degree_type text, institution text, certification_date date);"

echo "Creating tables..."

# Create Table 1
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS organizations_by_update_time (
    time_bucket text,
    updated_at timestamp,
    org_id uuid,
    org_type text,
    org_name text,
    org_address text,
    region_id uuid,
    region_name text,
    country_id uuid,
    country_name text,
    PRIMARY KEY (time_bucket, updated_at, org_id)
) WITH CLUSTERING ORDER BY (updated_at DESC, org_id ASC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '1', 
                   'compaction_window_unit': 'DAYS'};"

# Create Table 2
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS customers (
    cust_id uuid,
    cust_name frozen<person_name>,
    joined_date timestamp,
    country_id uuid,
    country_name text,  
    state_id uuid,
    state_name text,   
    region_id uuid,   
    region_name text,
    PRIMARY KEY (cust_id)
) WITH compaction = {'class': 'SizeTieredCompactionStrategy'};"

# Create Table 3
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS records_by_account_and_amount (
    account_id uuid,
    amount_bucket int,
    medical_amt decimal,
    medical_id uuid,
    tx_date timestamp,
    medical_purpose text,
    doc_id uuid,
    doc_name frozen<person_name>,
    org_id uuid,
    org_name text,
    cust_id uuid,
    cust_name frozen<person_name>,
    PRIMARY KEY (account_id, amount_bucket, medical_amt, medical_id)
) WITH CLUSTERING ORDER BY (amount_bucket DESC, medical_amt DESC, medical_id ASC)
  AND compaction = {'class': 'SizeTieredCompactionStrategy'};"

# Create Table 4a
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS doctors_by_popularity (
    count_bucket int,
    doc_id uuid,
    doc_name frozen<person_name>,
    org_id uuid,
    org_name text,
    org_type text,
    experience_year int,
    credential frozen<doctor_credential>,
    PRIMARY KEY (count_bucket, doc_id)
) WITH CLUSTERING ORDER BY (doc_id ASC)
  AND compaction = {'class': 'SizeTieredCompactionStrategy'};"

# Create Table 4b
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS doctor_record_counts (
    count_bucket int,
    doc_id uuid,
    record_count counter,
    PRIMARY KEY (count_bucket, doc_id)
) WITH CLUSTERING ORDER BY (doc_id ASC)
  AND compaction = {'class': 'SizeTieredCompactionStrategy'};"

# Create Table 5
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS monthly_expenses_by_account (
    account_id uuid,
    year_month text,
    total_medical_amt decimal,
    transaction_count int,
    cust_id uuid,
    cust_name frozen<person_name>,
    PRIMARY KEY (account_id, year_month)
) WITH CLUSTERING ORDER BY (year_month DESC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '30', 
                   'compaction_window_unit': 'DAYS'};"

# Create Table 6
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS upcoming_appointments_by_account (
    account_id uuid,
    tx_time timestamp,
    medical_id uuid,
    doc_id uuid,
    doc_name frozen<person_name>,
    medical_purpose text,
    org_id uuid,
    org_name text,
    org_address text,
    cust_id uuid,
    cust_name frozen<person_name>,
    PRIMARY KEY (account_id, tx_time, medical_id)
) WITH CLUSTERING ORDER BY (tx_time ASC, medical_id ASC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '7', 
                   'compaction_window_unit': 'DAYS'};"

# Create Table 7
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS doctors (
    doc_id uuid,
    org_id uuid,
    org_name text,
    org_type text,
    doc_name frozen<person_name>,
    experience_year int,
    birth_year int,
    credential frozen<doctor_credential>,
    PRIMARY KEY (doc_id)
) WITH compaction = {'class': 'SizeTieredCompactionStrategy'};"

# Create Table 8
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS records_by_doctor_and_date (
    doc_id uuid,
    tx_time timestamp,
    medical_id uuid,
    account_id uuid,
    medical_purpose text,
    medical_amt decimal,
    doc_name frozen<person_name>,
    org_id uuid,
    org_name text,
    cust_id uuid,
    cust_name frozen<person_name>,
    PRIMARY KEY (doc_id, tx_time, medical_id)
) WITH CLUSTERING ORDER BY (tx_time DESC, medical_id ASC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '30', 
                   'compaction_window_unit': 'DAYS'};"

# Create Table 9a
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS organization_ratings (
    org_id uuid,
    org_name text,
    org_type text,
    org_address text,
    region_id uuid,
    region_name text,
    avg_rating decimal,
    PRIMARY KEY (org_id)
) WITH compaction = {'class': 'SizeTieredCompactionStrategy'};"

# Create Table 10
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS upcoming_appointments_by_doctor (
    doc_id uuid,
    tx_time timestamp,
    medical_id uuid,
    account_id uuid,
    medical_purpose text,
    doc_name frozen<person_name>,
    org_id uuid,
    org_name text,
    cust_id uuid,
    cust_name frozen<person_name>,
    PRIMARY KEY (doc_id, tx_time, medical_id)
) WITH CLUSTERING ORDER BY (tx_time ASC, medical_id ASC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '7', 
                   'compaction_window_unit': 'DAYS'};"

# Create Table 11
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS accounts_by_created_time (
    time_bucket text,
    created_at timestamp,
    account_id uuid,
    cust_id uuid,
    cust_name frozen<person_name>,
    address text,
    country_id uuid,
    country_name text,
    state_id uuid,
    state_name text,
    region_id uuid,
    region_name text,
    PRIMARY KEY (time_bucket, created_at, account_id)
) WITH CLUSTERING ORDER BY (created_at DESC, account_id ASC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '1', 
                   'compaction_window_unit': 'DAYS'};"

# Create Table 12a
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS daily_org_transactions (
    org_id uuid,
    day date,
    org_name text,
    org_type text,
    total_revenue decimal,
    PRIMARY KEY (org_id, day)
) WITH CLUSTERING ORDER BY (day DESC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '30', 
                   'compaction_window_unit': 'DAYS'};"

# Create Table 12b
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS daily_org_transaction_counts (
    org_id uuid,
    day date,
    transaction_count counter,
    PRIMARY KEY (org_id, day)
) WITH CLUSTERING ORDER BY (day DESC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy', 
                   'compaction_window_size': '30', 
                   'compaction_window_unit': 'DAYS'};"

# Create Table 13a
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS regions_by_popularity (
    count_bucket int,
    region_id uuid,
    region_name text,
    country_id uuid,
    country_name text,
    total_revenue decimal,
    PRIMARY KEY (count_bucket, region_id)
) WITH CLUSTERING ORDER BY (region_id ASC)
  AND compaction = {'class': 'SizeTieredCompactionStrategy'};"

# Create Table 13b
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS region_customer_counts (
    count_bucket int,
    region_id uuid,
    customer_count counter,
    PRIMARY KEY (count_bucket, region_id)
) WITH CLUSTERING ORDER BY (region_id ASC)
  AND compaction = {'class': 'SizeTieredCompactionStrategy'};"

# Create Table 14
cqlsh -e "USE $KEYSPACE_NAME; CREATE TABLE IF NOT EXISTS revenue_by_doctor_and_org (
    org_id uuid,
    revenue_bucket int,
    total_revenue decimal,
    doc_id uuid,
    org_name text,
    org_type text,
    doc_name frozen<person_name>,
    experience_year int,
    transaction_count int,
    PRIMARY KEY (org_id, revenue_bucket, total_revenue, doc_id)
) WITH CLUSTERING ORDER BY (revenue_bucket DESC, total_revenue DESC, doc_id ASC)
  AND compaction = {'class': 'SizeTieredCompactionStrategy'};"

echo "Done! All tables have been created."

echo "Do you want to load the data into the tables? (y/n)"
read answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
  ./load_data.sh
else
  echo "Cancelled"
fi
