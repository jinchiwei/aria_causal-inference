#!/usr/bin/env python3
"""
Combine ARIA labeling results from a specific date run into a single CSV file.

Usage:
    python combine_run_results.py --date_run 2024-01-15_143022
    python combine_run_results.py --date_run 2024-01-15_143022 --config ../config.yaml
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def combine_results(date_run, config):
    """Combine individual model results from a specific date run"""
    base_dir = Path(config['data']['base_output_dir'])
    run_dir = base_dir / date_run
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return False
    
    # Find all CSV files in the run directory
    csv_files = list(run_dir.glob("aria_labels_*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {run_dir}")
        return False
    
    print(f"Found {len(csv_files)} CSV files to combine")
    
    # Load the first file to get accession numbers
    first_df = pd.read_csv(csv_files[0])
    combined_data = {"Accession Number": first_df["Accession Number"]}
    
    # Process each CSV file
    for csv_file in sorted(csv_files):
        print(f"Processing: {csv_file.name}")
        
        # Load the CSV
        df = pd.read_csv(csv_file)
        
        # Add columns from this model
        for col in df.columns:
            if col != "Accession Number":
                combined_data[col] = df[col]
    
    # Create final DataFrame
    final_df = pd.DataFrame(combined_data)
    
    # Reorder columns for better readability
    ordered_columns = ["Accession Number"]
    
    # Add condition columns for each model
    for condition in config['conditions']:
        for model in config['models']:
            model_underscore = model.replace("-", "_")
            col_name = f"{condition}_{model_underscore}"
            if col_name in final_df.columns:
                ordered_columns.append(col_name)
    
    # Add reasoning columns
    for condition in ["aria_e", "aria_h"]:
        for model in config['models']:
            model_underscore = model.replace("-", "_")
            col_name = f"reasoning_{condition}_{model_underscore}"
            if col_name in final_df.columns:
                ordered_columns.append(col_name)
    
    # Reorder DataFrame
    available_columns = [col for col in ordered_columns if col in final_df.columns]
    final_df = final_df[available_columns]
    
    # Save combined results
    output_file = run_dir / "aria_labels_combined.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"Combined results saved to: {output_file}")
    print(f"Final DataFrame shape: {final_df.shape}")
    print(f"Total columns: {len(final_df.columns)}")
    
    # Show summary
    print("\nSummary by condition:")
    for condition in config['conditions']:
        condition_cols = [col for col in final_df.columns if col.startswith(f"{condition}_")]
        if condition_cols:
            print(f"  {condition}: {len(condition_cols)} model columns")
            
            # Show positive counts across models
            positive_counts = []
            for col in condition_cols:
                count = (final_df[col] == 1).sum()
                positive_counts.append(count)
            
            if positive_counts:
                min_pos = min(positive_counts)
                max_pos = max(positive_counts)
                avg_pos = sum(positive_counts) / len(positive_counts)
                print(f"    Positive cases: min={min_pos}, max={max_pos}, avg={avg_pos:.1f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Combine ARIA labeling results from a specific date run")
    parser.add_argument("--date_run", required=True, help="Date run identifier (e.g., 2024-01-15_143022)")
    parser.add_argument("--config", default="../config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Combine results
    success = combine_results(args.date_run, config)
    
    if success:
        print("Results combined successfully!")
        return 0
    else:
        print("Failed to combine results")
        return 1


if __name__ == "__main__":
    exit(main())
