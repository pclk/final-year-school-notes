#!/bin/bash

# Set variables
KEYSPACE_NAME="wearedoctors"
CSV_DIR=${CSV_DIR:-"./data"} # Use the directory where temp.sh outputs files

# Load data from CSV files
echo "Loading data from CSV files..."

# Check if CSV directory exists
if [ ! -d "$CSV_DIR" ]; then
  echo "CSV directory $CSV_DIR not found!"
  echo "Please run temp.sh first to generate the split CSV files."
  exit 1
fi

# Define required files including split counter table files
required_files=(
  "organizations_by_update_time.csv"
  "customers.csv"
  "records_by_account_and_amount.csv"
  "doctors_by_popularity.csv" # New split file
  "doctor_record_counts.csv"  # New counter file
  "monthly_expenses_by_account.csv"
  "upcoming_appointments_by_account.csv"
  "doctors.csv"
  "records_by_doctor_and_date.csv"
  "organization_ratings.csv" # New split file
  "upcoming_appointments_by_doctor.csv"
  "accounts_by_created_time.csv"
  "daily_org_transactions.csv"       # New split file
  "daily_org_transaction_counts.csv" # New counter file
  "regions_by_popularity.csv"        # New split file
  "region_customer_counts.csv"       # New counter file
  "revenue_by_doctor_and_org.csv"
)

# Check if all required files exist
for file in "${required_files[@]}"; do
  if [ ! -f "$CSV_DIR/$file" ]; then
    echo "Required file $CSV_DIR/$file not found!"
    exit 1
  fi
done

echo "Importing data to Cassandra..."

# Import regular tables (not counter tables)
# Table 1: organizations_by_update_time
cqlsh -e "USE $KEYSPACE_NAME; COPY organizations_by_update_time (time_bucket, updated_at, org_id, org_type, org_name, org_address, region_id, region_name, country_id, country_name) FROM '$CSV_DIR/organizations_by_update_time.csv' WITH HEADER = true;"

# Table 2: customers
cqlsh -e "USE $KEYSPACE_NAME; COPY customers (cust_id, cust_name, joined_date, country_id, country_name, state_id, state_name, region_id, region_name) FROM '$CSV_DIR/customers.csv' WITH HEADER = true;"

# Table 3: records_by_account_and_amount
cqlsh -e "USE $KEYSPACE_NAME; COPY records_by_account_and_amount (account_id, amount_bucket, medical_amt, medical_id, tx_date, medical_purpose, doc_id, doc_name, org_id, org_name, cust_id, cust_name) FROM '$CSV_DIR/records_by_account_and_amount.csv' WITH HEADER = true;"

# Table 4a: doctors_by_popularity (non-counter part)
cqlsh -e "USE $KEYSPACE_NAME; COPY doctors_by_popularity (count_bucket, doc_id, doc_name, org_id, org_name, org_type, experience_year, credential) FROM '$CSV_DIR/doctors_by_popularity.csv' WITH HEADER = true;"

# Table 4b: doctor_record_counts (counter part)
echo "Setting up counter data for doctor_record_counts..."
# Process each line of the counter CSV to create UPDATE statements
awk -F, 'NR>1 {print "UPDATE '$KEYSPACE_NAME'.doctor_record_counts SET record_count = record_count + "$3" WHERE count_bucket = "$1" AND doc_id = "$2";"}' "$CSV_DIR/doctor_record_counts.csv" >/tmp/doctor_record_counts.cql
cqlsh -f /tmp/doctor_record_counts.cql

# Table 5: monthly_expenses_by_account
cqlsh -e "USE $KEYSPACE_NAME; COPY monthly_expenses_by_account (account_id, year_month, total_medical_amt, transaction_count, cust_id, cust_name) FROM '$CSV_DIR/monthly_expenses_by_account.csv' WITH HEADER = true;"

# Table 6: upcoming_appointments_by_account
cqlsh -e "USE $KEYSPACE_NAME; COPY upcoming_appointments_by_account (account_id, tx_time, medical_id, doc_id, doc_name, medical_purpose, org_id, org_name, org_address, cust_id, cust_name) FROM '$CSV_DIR/upcoming_appointments_by_account.csv' WITH HEADER = true;"

# Table 7: doctors
cqlsh -e "USE $KEYSPACE_NAME; COPY doctors (doc_id, org_id, org_name, org_type, doc_name, experience_year, birth_year, credential) FROM '$CSV_DIR/doctors.csv' WITH HEADER = true;"

# Table 8: records_by_doctor_and_date
cqlsh -e "USE $KEYSPACE_NAME; COPY records_by_doctor_and_date (doc_id,  tx_time, medical_id, account_id, medical_purpose, medical_amt, doc_name, org_id, org_name, cust_id, cust_name) FROM '$CSV_DIR/records_by_doctor_and_date.csv' WITH HEADER = true;"

# Table 9a: organization_ratings (non-counter part)
cqlsh -e "USE $KEYSPACE_NAME; COPY organization_ratings (org_id, org_name, org_type, org_address, region_id, region_name, avg_rating) FROM '$CSV_DIR/organization_ratings.csv' WITH HEADER = true;"

# Table 10: upcoming_appointments_by_doctor
cqlsh -e "USE $KEYSPACE_NAME; COPY upcoming_appointments_by_doctor (doc_id, tx_time, medical_id, account_id, medical_purpose, doc_name, org_id, org_name, cust_id, cust_name) FROM '$CSV_DIR/upcoming_appointments_by_doctor.csv' WITH HEADER = true;"

# Table 11: accounts_by_created_time
cqlsh -e "USE $KEYSPACE_NAME; COPY accounts_by_created_time (time_bucket, created_at, account_id, cust_id, cust_name, address, country_id, country_name, state_id, state_name, region_id, region_name) FROM '$CSV_DIR/accounts_by_created_time.csv' WITH HEADER = true;"

# Table 12a: daily_org_transactions (non-counter part)
cqlsh -e "USE $KEYSPACE_NAME; COPY daily_org_transactions (org_id, day, org_name, org_type, total_revenue) FROM '$CSV_DIR/daily_org_transactions.csv' WITH HEADER = true;"

# Table 12b: daily_org_transaction_counts (counter part)
echo "Setting up counter data for daily_org_transaction_counts..."
>/tmp/daily_org_transaction_counts.cql
tail -n +2 "$CSV_DIR/daily_org_transaction_counts.csv" | while IFS=, read -r org_id day transaction_count remainder; do
  echo "UPDATE $KEYSPACE_NAME.daily_org_transaction_counts SET transaction_count = transaction_count + $transaction_count WHERE org_id = $org_id AND day = '$day';" >>/tmp/daily_org_transaction_counts.cql
done
cqlsh -f /tmp/daily_org_transaction_counts.cql

# Table 13a: regions_by_popularity (non-counter part)
cqlsh -e "USE $KEYSPACE_NAME; COPY regions_by_popularity (count_bucket, region_id, region_name, country_id, country_name, total_revenue) FROM '$CSV_DIR/regions_by_popularity.csv' WITH HEADER = true;"

# Table 13b: region_customer_counts (counter part)
echo "Setting up counter data for region_customer_counts..."
awk -F, 'NR>1 {print "UPDATE '$KEYSPACE_NAME'.region_customer_counts SET customer_count = customer_count + "$3" WHERE count_bucket = "$1" AND region_id = "$2";"}' "$CSV_DIR/region_customer_counts.csv" >/tmp/region_customer_counts.cql
cqlsh -f /tmp/region_customer_counts.cql

# Table 14: revenue_by_doctor_and_org
cqlsh -e "USE $KEYSPACE_NAME; COPY revenue_by_doctor_and_org (org_id, revenue_bucket, total_revenue, doc_id, org_name, org_type, doc_name, experience_year, transaction_count) FROM '$CSV_DIR/revenue_by_doctor_and_org.csv' WITH HEADER = true;"

echo "Setup complete! All tables have been populated with data."
echo "You can now run your select statements."
echo "Do you want to run automated select queries on the tables? (y/n)"
read answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
  ./select.sh
else
  echo "Cancelled"
fi
