#!/bin/bash

# AIR Download Script
# Usage: ./run_air-download.sh [accessions_file] [output_directory]
# Example: ./run_air-download.sh (uses default accessions file)
# Example: ./run_air-download.sh custom_accessions.txt /data/output

# Set default values
DEFAULT_ACCESSIONS="/data/rauschecker1/jkw/projects/alzheimers/aria/aria_prediction_modeling/data/2025-07-08_aria-accessions.csv"
ACCESSIONS_FILE="${1:-$DEFAULT_ACCESSIONS}"
OUTPUT_DIR="${2:-/data/rauschecker1/jkw/projects/alzheimers/aria/aria_prediction_modeling/data/ucsf_aria}"
# PROJECT_ID="3"
# PROFILE="72"
# SERIES_FILTER="t1,spgr,bravo,mpr"

# Check if accessions file exists
if [ ! -f "$ACCESSIONS_FILE" ]; then
    echo "Error: Accessions file '$ACCESSIONS_FILE' not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the air-download.py script
echo "Starting AIR download..."
echo "Accessions file: $ACCESSIONS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Project ID: $PROJECT_ID"
echo "Profile: $PROFILE"
echo "Series filter: $SERIES_FILTER"
echo ""

python run_air-download.py "$ACCESSIONS_FILE" \
    -o "$OUTPUT_DIR" \
    -pj "$PROJECT_ID"
    # -pf "$PROFILE" \
    # -s "$SERIES_FILTER"

echo "AIR download completed!" 