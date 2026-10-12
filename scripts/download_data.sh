#!/usr/bin/env bash
# Download the open data this project uses.
#   1. Synthea synthetic patient records (CSV export). No real patients, no PHI.
#   2. Public CMS medical policy documents (you add these; see below).
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR/synthea" "$RAW_DIR/policies"

# --- 1. Synthea sample data (~1,000 synthetic patients, CSV format) -------------
SYNTHEA_URL="https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip"
if [ ! -d "$RAW_DIR/synthea/csv" ]; then
  echo "Downloading Synthea sample data..."
  curl -L --fail -o "$RAW_DIR/synthea/synthea_csv.zip" "$SYNTHEA_URL"
  unzip -q -o "$RAW_DIR/synthea/synthea_csv.zip" -d "$RAW_DIR/synthea/tmp"
  # The archive nests the CSVs one level down; flatten into synthea/csv.
  mkdir -p "$RAW_DIR/synthea/csv"
  find "$RAW_DIR/synthea/tmp" -name "*.csv" -exec mv {} "$RAW_DIR/synthea/csv/" \;
  rm -rf "$RAW_DIR/synthea/tmp" "$RAW_DIR/synthea/synthea_csv.zip"
  echo "Synthea CSVs in $RAW_DIR/synthea/csv"
else
  echo "Synthea data already present."
fi
# For more patients, generate your own with https://github.com/synthetichealth/synthea
#   ./run_synthea -p 5000 --exporter.csv.export=true

# --- 2. CMS medical policy documents -------------------------------------------
# Download a handful of public National Coverage Determinations (NCDs) and Local
# Coverage Determinations (LCDs) as PDF from the CMS Medicare Coverage Database:
#   https://www.cms.gov/medicare-coverage-database/search.aspx
# Save them into data/raw/policies/. Plain .txt or .md files work too.
echo
echo "Next: save CMS coverage policy PDFs into $RAW_DIR/policies/"
echo "      (Medicare Coverage Database: https://www.cms.gov/medicare-coverage-database/)"
