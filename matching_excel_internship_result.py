import pandas as pd
import os

# Get filename from user
filename = input("Enter Excel file name (with .xlsx): ").strip()

# Path to Downloads folder
downloads_path = os.path.expanduser("~/Downloads")

# Full path to the input file
input_file = os.path.join(downloads_path, filename)

# Check file exists
if not os.path.exists(input_file):
    print("❌ File not found in Downloads folder!")
    exit()

# Read Excel
df = pd.read_excel(input_file)

print("Columns found:", list(df.columns))

# Ask user which two columns to compare
col1 = input("Enter first column name to compare: ").strip()
col2 = input("Enter second column name to compare: ").strip()

# Validate columns
if col1 not in df.columns or col2 not in df.columns:
    print("❌ One or both column names are invalid!")
    exit()

# Compare columns and extract matched rows
matched_rows = df[df[col1] == df[col2]]

# Create output file name
output_file = os.path.join(downloads_path, "matched_output.xlsx")

# Save result
matched_rows.to_excel(output_file, index=False)

print(f"✅ Output file generated: {output_file}")
print(f"✅ Total matched rows: {len(matched_rows)}")

