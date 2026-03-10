#!/usr/bin/env python3
"""
ARIA Label Generation Script

This script generates binary labels for ARIA-E, ARIA-H, and related conditions
using various AI models with chain-of-thought reasoning.

Usage:
    python versa_labels.py --model <model_name> [--output_dir <path>] [--verbose]

Example:
    python versa_labels.py --model gpt-4o-2024-08-06
"""

import argparse
import importlib
import sys
import re
import yaml
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add utils to path
sys.path.append('/data/rauschecker2/jkw/aria/dev/src/utils')
import versa_api


def load_config(config_path="/data/rauschecker2/jkw/aria/dev/src/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Define prompts for each condition
PROMPTS = {
    "aria_e": """Given the following radiology report, does the patient have ARIA-E (Amyloid-Related Imaging Abnormalities - Edema)?

Please provide your reasoning step-by-step:
1. First, identify any mentions of edema, fluid accumulation, vasogenic edema, or FLAIR hyperintensity
2. Consider the context and location of any edema findings
3. Determine if the findings are consistent with ARIA-E (amyloid-related imaging abnormalities with edema)
4. Make your final determination

Format your response as:
REASONING: [Your step-by-step analysis]
ANSWER: [1 if ARIA-E is present, 0 if not present]

Report: 
""",
    
    "aria_h": """Given the following radiology report, does the patient have ARIA-H (Amyloid-Related Imaging Abnormalities - Hemorrhage)?

Please provide your reasoning step-by-step:
1. First, identify any mentions of hemorrhage, microhemorrhage, microbleeds, hemosiderin deposits, or susceptibility artifacts
2. Consider the pattern, distribution, and characteristics of any hemorrhagic findings
3. Determine if the findings are consistent with ARIA-H (amyloid-related imaging abnormalities with hemorrhage)
4. Make your final determination

Format your response as:
REASONING: [Your step-by-step analysis]
ANSWER: [1 if ARIA-H is present, 0 if not present]

Report: 
""",
    
    "edema": """Given the following radiology report, does the patient have brain edema?

Your response should be exactly one of the following: "1" (if edema is present) or "0" (if edema is not present)

Report: 
""",
    
    "effusion": """Given the following radiology report, does the patient have effusion?

Your response should be exactly one of the following: "1" (if effusion is present) or "0" (if effusion is not present)

Report: 
""",
    
    "microhemorrhage": """Given the following radiology report, how many microhemorrhages (also called microbleeds or cerebral microhemorrhages) does the patient have?

Your response should be exactly one of the following:
- "0" if the patient has 0-4 microhemorrhages (none or few)
- "1" if the patient has 5-9 microhemorrhages
- "2" if the patient has 10 or more microhemorrhages

If the report mentions microhemorrhages but does not specify a count, use your best judgment based on the language (e.g., "few" or "scattered" suggests 0-4, "numerous" or "innumerable" suggests 10+).

Report:
""",
    
    "superficial_siderosis": """Given the following radiology report, does the patient have superficial siderosis?

Your response should be exactly one of the following: "1" (if superficial siderosis is present) or "0" (if superficial siderosis is not present)

Report: 
"""
}


def parse_chain_of_thought_response(response, condition):
    """Parse chain-of-thought response to extract reasoning and answer"""
    if not isinstance(response, str):
        return 0, "No response provided"
    
    response = response.strip()
    reasoning = ""
    answer = 0
    
    # Try to extract REASONING and ANSWER sections
    if "REASONING:" in response and "ANSWER:" in response:
        parts = response.split("ANSWER:")
        if len(parts) >= 2:
            reasoning_part = parts[0].replace("REASONING:", "").strip()
            answer_part = parts[1].strip()
            
            reasoning = reasoning_part
            
            # Extract the answer (should be 0 or 1)
            try:
                # Look for 0 or 1 in the answer part
                answer_clean = answer_part.replace('"', '').replace("'", "").strip()
                if answer_clean.startswith("1"):
                    answer = 1
                elif answer_clean.startswith("0"):
                    answer = 0
                else:
                    # Try to find 0 or 1 anywhere in the answer
                    numbers = re.findall(r'\b[01]\b', answer_clean)
                    if numbers:
                        answer = int(numbers[0])
            except:
                answer = 0
    else:
        # Fallback: try to extract any reasoning and look for 0/1
        reasoning = response
        try:
            numbers = re.findall(r'\b[01]\b', response)
            if numbers:
                answer = int(numbers[-1])  # Take the last 0 or 1 found
        except:
            answer = 0
    
    return answer, reasoning


def parse_simple_response(response, valid_values=None):
    """Parse simple response (for non-ARIA conditions)"""
    if valid_values is None:
        valid_values = [0, 1]

    if not isinstance(response, str):
        return 0

    response = response.replace('"', '').replace("'", "").strip()
    try:
        answer = int(response)
        if answer not in valid_values:
            answer = 0
    except ValueError:
        # Try to find valid values in the response
        pattern = r'\b[' + ''.join(str(v) for v in valid_values) + r']\b'
        numbers = re.findall(pattern, response)
        if numbers:
            answer = int(numbers[-1])
        else:
            answer = 0

    return answer


def load_data(data_path):
    """Load the ARIA data from Excel file"""
    aria_data_df = pd.read_excel(data_path)
    accession_to_report = dict(zip(aria_data_df["Accession Number"], aria_data_df["Report Text"]))
    return accession_to_report


def generate_labels_for_model(model_name, accession_to_report, conditions, output_dir, verbose=False):
    """Generate labels for all conditions using a specific model"""
    print(f"Generating labels using model: {model_name}")
    
    # Initialize VersaAI
    versa_ai = versa_api.VersaAI(
        deployment=model_name,
        usage_log_dir=output_dir,
        run_dir=output_dir,
    )
    
    # Storage for results
    results = {}
    reasoning_storage = {}
    
    # Initialize storage
    for condition in conditions:
        results[condition] = {}
        if condition in ["aria_e", "aria_h"]:
            reasoning_storage[condition] = {}
    
    # Process each condition
    for condition in conditions:
        print(f"Processing {condition} with {model_name}...")
        
        for accession, report in tqdm(accession_to_report.items(), 
                                    desc=f"Labeling {condition}"):
            prompt = PROMPTS[condition] + str(report)
            
            try:
                response = versa_ai.predict(
                    prompt,
                    verbose=verbose,
                    request_metadata={
                        "condition": condition,
                        "accession": accession,
                    },
                )
                
                # Parse response based on condition type
                if condition in ["aria_e", "aria_h"]:
                    # Chain-of-thought parsing for ARIA conditions
                    label, reasoning = parse_chain_of_thought_response(response, condition)
                    reasoning_storage[condition][accession] = reasoning
                else:
                    # Simple parsing for other conditions
                    if condition == "microhemorrhage":
                        label = parse_simple_response(response, valid_values=[0, 1, 2])
                    else:
                        label = parse_simple_response(response)
                
                results[condition][accession] = label
                
            except Exception as e:
                print(f"Error processing {accession} for {condition}: {e}")
                results[condition][accession] = 0
                if condition in ["aria_e", "aria_h"]:
                    reasoning_storage[condition][accession] = f"Error: {str(e)}"
    
    return results, reasoning_storage


def create_output_dataframe(accession_to_report, results, reasoning_storage, model_name):
    """Create the final output DataFrame"""
    all_accessions = list(accession_to_report.keys())
    
    # Start with accession numbers
    data = {"Accession Number": all_accessions}
    
    # Add binary labels for all conditions
    for condition in results.keys():
        data[f"{condition}_{model_name}"] = [
            results[condition].get(acc, 0) for acc in all_accessions
        ]
    
    # Add reasoning columns for ARIA-E and ARIA-H
    for condition in ["aria_e", "aria_h"]:
        if condition in reasoning_storage:
            data[f"reasoning_{condition}_{model_name}"] = [
                reasoning_storage[condition].get(acc, "") for acc in all_accessions
            ]
    
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Generate ARIA labels using AI models")
    parser.add_argument("--model", required=True, help="AI model to use for labeling")
    parser.add_argument("--config", default="/data/rauschecker2/jkw/aria/dev/src/config.yaml", help="Path to config file")
    parser.add_argument("--date_run", help="Date run identifier (YYYY-MM-DD_HHMMSS)")
    parser.add_argument("--conditions", nargs="+", help="Specific conditions to label (default: all from config)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate model
    if args.model not in config['models']:
        print(f"Error: Model '{args.model}' not found in config. Available models: {config['models']}")
        return 1
    
    # Create date-specific output directory
    if args.date_run:
        date_run = args.date_run
    else:
        date_run = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    output_dir = Path(config['data']['base_output_dir']) / date_run
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which conditions to run
    conditions = args.conditions if args.conditions else config['conditions']

    print(f"Run date: {date_run}")
    print(f"Output directory: {output_dir}")
    print(f"Conditions: {conditions}")
    print(f"Loading data from: {config['data']['input_file']}")

    # Load data
    accession_to_report = load_data(config['data']['input_file'])
    print(f"Loaded {len(accession_to_report)} reports")

    # Generate labels
    results, reasoning_storage = generate_labels_for_model(
        args.model, accession_to_report, conditions, output_dir, args.verbose
    )

    # Create output DataFrame
    df = create_output_dataframe(accession_to_report, results, reasoning_storage, args.model)

    # Save to CSV - merge with existing if only running subset of conditions
    output_file = output_dir / f"aria_labels_{args.model.replace('-', '_')}.csv"
    if output_file.exists() and args.conditions:
        existing_df = pd.read_csv(output_file)
        # Update columns from new run, keep the rest
        for col in df.columns:
            if col != "Accession Number":
                existing_df[col] = df[col]
        df = existing_df
        print(f"Merged with existing CSV (updated: {[c for c in df.columns if any(cond in c for cond in conditions)]})")

    df.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Show summary statistics
    print("\nLabel counts:")
    for condition in conditions:
        col_name = f"{condition}_{args.model}"
        if col_name in df.columns:
            counts = df[col_name].value_counts().sort_index()
            print(f"  {condition}: {dict(counts)}")
    
    return 0


if __name__ == "__main__":
    main()
