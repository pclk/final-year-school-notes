#!/bin/bash

# This script fixes formatting issues in CSV files before loading into Cassandra
CSV_DIR=${CSV_DIR:-"./new_csv_data"}
FIXED_DIR="./fixed_csv_data"

# Create directory for fixed files
mkdir -p "$FIXED_DIR"

echo "Fixing CSV files formatting..."

# Function to fix UUID and UDT formats in a file
fix_file() {
    input_file="$1"
    output_file="$2"
    
    # Copy header
    head -1 "$input_file" > "$output_file"
    
    # Process data rows - replace example UUIDs with valid UUIDs and fix UDT format
    tail -n +2 "$input_file" | sed -E '
        # Fix UUIDs - replace with valid hexadecimal pattern
        s/[a-z][0-9][a-z][0-9][a-z][0-9][a-z][0-9]-[a-z][0-9][a-z][0-9]-[0-9]{4}-[0-9]{4}-[0-9a-z]{12}/12345678-90ab-cdef-1234-567890abcdef/g;
        s/[a-z][0-9][a-z][0-9][a-z][0-9][a-z][0-9]/12345678/g;
        
        # Fix UDT format - change single quotes to double quotes
        s/{first_name:'\''([^'\'']*)'\',last_name:'\''([^'\'']*)\'\'}/{"first_name":"\1","last_name":"\2"}/g;
    ' >> "$output_file"
    
    echo "Fixed $input_file -> $output_file"
}

# Process each CSV file
for file in "$CSV_DIR"/*.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        fix_file "$file" "$FIXED_DIR/$filename"
    fi
done

echo "All files fixed. Fixed files are in $FIXED_DIR"
