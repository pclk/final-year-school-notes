#!/usr/bin/env python3
import os
import csv
import re
import sys
from datetime import datetime


def process_csv_file(input_file, output_file):
    """Process a CSV file fixing timestamp issues and removing tx_date column"""
    with open(input_file, "r", newline="", encoding="utf-8") as infile:
        try:
            reader = csv.reader(infile)
            header = next(reader)
            data = list(reader)
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            return False

    # Check for timestamp fields that may need special handling
    tx_date_idx = -1
    tx_time_idx = -1

    for i, col in enumerate(header):
        if col.lower() == "tx_date":
            tx_date_idx = i
        elif col.lower() == "tx_time":
            tx_time_idx = i

    # If we don't have both columns, just copy the file
    if tx_date_idx == -1 or tx_time_idx == -1:
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            writer.writerows(data)
        print(f"No timestamp columns found in {input_file}, copied as is")
        return True

    # Create new header without tx_date
    new_header = [col for i, col in enumerate(header) if i != tx_date_idx]

    # Fix the data
    fixed_data = []

    for row in data:
        # Skip rows that are too short
        if len(row) < len(header):
            print(f"Warning: Skipping short row in {input_file}")
            continue

        # Fix timestamp by combining date and time
        date_val = row[tx_date_idx] if tx_date_idx < len(row) else ""
        time_val = row[tx_time_idx] if tx_time_idx < len(row) else ""

        # Create combined timestamp
        if re.match(r"^\d{2}:\d{2}:\d{2}Z?$", time_val) and re.match(
            r"^\d{4}-\d{2}-\d{2}$", date_val
        ):
            row[tx_time_idx] = f"{date_val}T{time_val}"

        # Create new row without tx_date
        new_row = [val for i, val in enumerate(row) if i != tx_date_idx]
        fixed_data.append(new_row)

    # Write fixed data
    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(new_header)
        writer.writerows(fixed_data)

    print(f"Fixed timestamps and removed tx_date column in {output_file}")
    return True


def main():
    input_dir = "new_csv_data_fixed"
    output_dir = "data"

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print(f"Processing files from {input_dir} to {output_dir}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Process each CSV file
    success_count = 0
    error_count = 0

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist!")
        sys.exit(1)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if process_csv_file(input_path, output_path):
                success_count += 1
            else:
                error_count += 1

    print(f"\nAll CSV files have been processed.")
    print(f"Successfully processed: {success_count} files")
    if error_count > 0:
        print(f"Errors encountered: {error_count} files")

    print("\nIMPORTANT: You need to modify your table schema and load_data.sh script!")
    print(
        "1. Remove tx_date column from table schemas that have both tx_date and tx_time"
    )
    print("2. Update your COPY commands in load_data.sh to exclude tx_date field")
    print(
        f'3. Use this directory in your load_data.sh: CSV_DIR=${{{output_dir}:-"{output_dir}"}}'
    )


if __name__ == "__main__":
    main()
