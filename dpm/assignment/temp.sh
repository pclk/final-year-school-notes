#!/bin/bash

CSV_DIR=${CSV_DIR:-"./data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./new_csv_data"}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Processing CSV files for new schema..."

# Process doctors_by_record_count.csv
echo "Processing doctors_by_record_count.csv..."
# 1. Create doctors_by_popularity.csv (all columns except record_count)
head -n 1 "$CSV_DIR/doctors_by_record_count.csv" | awk -F, '{gsub("record_count,", ""); print}' >"$OUTPUT_DIR/doctors_by_popularity.csv"
tail -n +2 "$CSV_DIR/doctors_by_record_count.csv" | awk -F, '{OFS=","; $2=""; gsub(",,", ","); print}' >>"$OUTPUT_DIR/doctors_by_popularity.csv"

# 2. Create doctor_record_counts.csv (just count_bucket, doc_id, record_count)
echo "count_bucket,doc_id,record_count" >"$OUTPUT_DIR/doctor_record_counts.csv"
tail -n +2 "$CSV_DIR/doctors_by_record_count.csv" | awk -F, '{print $1","$3","$2}' >>"$OUTPUT_DIR/doctor_record_counts.csv"

# Process organization_avg_ratings.csv
echo "Processing organization_avg_ratings.csv..."
# 1. Create organization_ratings.csv (all columns except rating_count)
head -n 1 "$CSV_DIR/organization_avg_ratings.csv" | awk -F, '{gsub(",rating_count", ""); print}' >"$OUTPUT_DIR/organization_ratings.csv"
tail -n +2 "$CSV_DIR/organization_avg_ratings.csv" | awk -F, '{OFS=","; NF--; print}' >>"$OUTPUT_DIR/organization_ratings.csv"

# 2. Create organization_rating_counts.csv (just org_id and rating_count)
echo "org_id,rating_count" >"$OUTPUT_DIR/organization_rating_counts.csv"
tail -n +2 "$CSV_DIR/organization_avg_ratings.csv" | awk -F, '{print $1","$8}' >>"$OUTPUT_DIR/organization_rating_counts.csv"

# Process daily_transactions_by_org.csv
echo "Processing daily_transactions_by_org.csv..."
# 1. Create daily_org_transactions.csv (non-counter data)
echo "org_id,org_name,org_type,day,total_revenue" >"$OUTPUT_DIR/daily_org_transactions.csv"
tail -n +2 "$CSV_DIR/daily_transactions_by_org.csv" | awk -F, '{print $1","$2","$3","$4","$9}' >>"$OUTPUT_DIR/daily_org_transactions.csv"

# 2. Create daily_org_transaction_counts.csv (counter data)
echo "org_id,day,transaction_count,emergency_count,routine_count,consultation_count" >"$OUTPUT_DIR/daily_org_transaction_counts.csv"
tail -n +2 "$CSV_DIR/daily_transactions_by_org.csv" | awk -F, '{print $1","$4","$5","$6","$7","$8}' >>"$OUTPUT_DIR/daily_org_transaction_counts.csv"

# Process regions_by_customer_count.csv
echo "Processing regions_by_customer_count.csv..."
# 1. Create regions_by_popularity.csv (all columns except customer_count)
head -n 1 "$CSV_DIR/regions_by_customer_count.csv" | awk -F, '{gsub("customer_count,", ""); print}' >"$OUTPUT_DIR/regions_by_popularity.csv"
tail -n +2 "$CSV_DIR/regions_by_customer_count.csv" | awk -F, '{OFS=","; $2=""; gsub(",,", ","); print}' >>"$OUTPUT_DIR/regions_by_popularity.csv"

# 2. Create region_customer_counts.csv (just count_bucket, region_id, customer_count)
echo "count_bucket,region_id,customer_count" >"$OUTPUT_DIR/region_customer_counts.csv"
tail -n +2 "$CSV_DIR/regions_by_customer_count.csv" | awk -F, '{print $1","$3","$2}' >>"$OUTPUT_DIR/region_customer_counts.csv"

# Copy all unchanged files
echo "Copying unchanged files..."
cp "$CSV_DIR/organizations_by_update_time.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/customers.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/records_by_account_and_amount.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/monthly_expenses_by_account.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/upcoming_appointments_by_account.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/doctors.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/records_by_doctor_and_date.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/upcoming_appointments_by_doctor.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/accounts_by_created_time.csv" "$OUTPUT_DIR/"
cp "$CSV_DIR/revenue_by_doctor_and_org.csv" "$OUTPUT_DIR/"

echo "CSV file processing complete. New files are in $OUTPUT_DIR"
