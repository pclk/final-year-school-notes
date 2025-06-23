#!/usr/bin/env python3
import os
import csv
import uuid
import re
import sys


def fix_uuid(uuid_str):
    """Make sure UUID is properly formatted with correct length and hex characters"""
    # If it already looks like a well-formed UUID, return it
    if re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        uuid_str.lower(),
    ):
        return uuid_str

    # Try to fix the UUID by ensuring each segment has correct length and only hex chars
    parts = uuid_str.split("-")
    if len(parts) != 5:
        return str(uuid.uuid4())  # Just create a new UUID if format is wrong

    required_lengths = [8, 4, 4, 4, 12]
    fixed_parts = []

    for i, (part, length) in enumerate(zip(parts, required_lengths)):
        # Convert non-hex chars to hex
        hex_part = ""
        for char in part:
            if char.lower() in "0123456789abcdef":
                hex_part += char.lower()
            else:
                hex_part += format(ord(char) % 16, "x")  # Convert to hex digit

        # Ensure correct length
        if len(hex_part) < length:
            hex_part = hex_part + "0" * (length - len(hex_part))  # Pad with zeros
        elif len(hex_part) > length:
            hex_part = hex_part[:length]  # Truncate

        fixed_parts.append(hex_part)

    return "-".join(fixed_parts)


def fix_udt(udt_str):
    """Fix formatting of UDT strings for Cassandra"""
    # Remove all layers of quotes
    udt_str = udt_str.strip()

    # Remove excessive quoting
    while (udt_str.startswith('"') and udt_str.endswith('"')) or (
        udt_str.startswith("'") and udt_str.endswith("'")
    ):
        udt_str = udt_str[1:-1].strip()

    # Extract UDT content if it looks like a UDT
    if udt_str.startswith("{") and udt_str.endswith("}"):
        # Extract the content between braces
        content = udt_str[1:-1].strip()

        # Fix field-value pairs
        parts = []
        for part in re.split(r",\s*", content):
            if ":" in part:
                field, value = part.split(":", 1)
                field = field.strip()
                value = value.strip()

                # Ensure value has proper quoting
                if value.startswith("'") and value.endswith("'"):
                    # Value already has single quotes, keep as is
                    pass
                elif value.startswith('"') and value.endswith('"'):
                    # Replace double quotes with single quotes
                    value = "'" + value[1:-1] + "'"
                elif not value.startswith("'"):
                    # Add single quotes if it's a string value and doesn't already have them
                    if not re.match(
                        r"^[0-9]+(\.[0-9]+)?$", value
                    ) and value.lower() not in ("true", "false", "null"):
                        value = "'" + value + "'"

                parts.append(f"{field}: {value}")

        # Reconstruct the UDT
        udt_str = "{" + ", ".join(parts) + "}"

    # Return the properly formatted UDT with quotation for CSV
    return f"{udt_str}"


def process_csv_file(input_file, output_file):
    """Process a CSV file fixing UUID and UDT issues"""
    with open(input_file, "r", newline="", encoding="utf-8") as infile:
        try:
            reader = csv.reader(infile)
            header = next(reader)
            data = list(reader)
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            return

    # Fix the data
    fixed_data = []
    for row in data:
        fixed_row = []
        for i, value in enumerate(row):
            # Check if it might be a UUID (contains hyphens in roughly UUID pattern)
            if re.search(
                r"[0-9a-zA-Z]{4,}-[0-9a-zA-Z]{2,}-[0-9a-zA-Z]{2,}-[0-9a-zA-Z]{2,}-[0-9a-zA-Z]{4,}",
                value,
            ):
                value = fix_uuid(value)
            # Check if it looks like a UDT (contains curly braces)
            elif (
                ("{" in value and "}" in value)
                or value.startswith('"""')
                and value.endswith('"""')
            ):
                value = fix_udt(value)
            fixed_row.append(value)
        fixed_data.append(fixed_row)

    # Write fixed data
    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(fixed_data)

    print(f"Processed {input_file} -> {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: ./fix_csv.py <csv_directory>")
        sys.exit(1)

    csv_dir = sys.argv[1]
    output_dir = f"{csv_dir}_fixed"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each CSV file
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(csv_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_csv_file(input_path, output_path)

    print(f"\nAll CSV files have been processed. Fixed files are in {output_dir}")
    print("To use the fixed files, modify your load_data.sh script:")


if __name__ == "__main__":
    main()
