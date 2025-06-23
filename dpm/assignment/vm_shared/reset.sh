#!/bin/bash

# Set variables
KEYSPACE_NAME="wearedoctors"

echo "Connecting to Cassandra..."
echo "Dropping all tables from keyspace '$KEYSPACE_NAME'..."

# Use the keyspace
cqlsh -e "USE $KEYSPACE_NAME;"

# Drop all tables
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS organizations_by_update_time;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS customers;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS records_by_account_and_amount;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS doctors_by_popularity;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS doctor_record_counts;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS monthly_expenses_by_account;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS upcoming_appointments_by_account;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS doctors;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS records_by_doctor_and_date;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS organization_ratings;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS organization_rating_counts;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS upcoming_appointments_by_doctor;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS accounts_by_created_time;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS daily_org_transactions;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS daily_org_transaction_counts;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS regions_by_popularity;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS region_customer_counts;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS revenue_by_doctor_and_org;"

# Drop old tables that might exist
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS doctors_by_record_count;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS organization_avg_ratings;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS daily_transactions_by_org;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TABLE IF EXISTS regions_by_customer_count;"

# Drop UDTs
cqlsh -e "USE $KEYSPACE_NAME; DROP TYPE IF EXISTS person_name;"
cqlsh -e "USE $KEYSPACE_NAME; DROP TYPE IF EXISTS doctor_credential;"

echo "All tables have been dropped."
echo "You can now run create_table.sh to recreate the tables with the new schema."

# Add this section if you want to immediately recreate the tables
echo "Do you want to recreate the tables now? (y/n)"
read answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
  ./create_table.sh
else
  echo "Tables not recreated. Run ./create_table.sh when you're ready."
fi
