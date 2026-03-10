#!/usr/bin/env python3
"""
Submit ARIA labeling jobs for all models in config.

This script:
1. Reads models from config.yaml
2. Submits one SLURM job per model
3. Exits immediately (doesn't wait)

Usage:
    python submit_all_jobs.py [--config <path>] [--dry_run]
"""

import argparse
import shutil
import subprocess
import yaml
from datetime import datetime
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def submit_job(model, date_run, config_path, conditions=None, dry_run=False):
    """Submit a SLURM job for a single model"""
    script_dir = Path(__file__).resolve().parent
    conditions_str = ",".join(conditions) if conditions else ""
    cmd = [
        "sbatch",
        str(script_dir / "run_aria_labels.sh"),
        model,
        date_run,
        str(config_path),
        conditions_str,
    ]
    
    if dry_run:
        print(f"DRY RUN: Would submit: {' '.join(cmd)}")
        return None
    else:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]  # Extract job ID
            print(f"Submitted {model}: Job ID {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for {model}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Submit ARIA labeling jobs for all models")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"), help="Path to config file")
    parser.add_argument("--conditions", nargs="+", help="Specific conditions to label (passed through to versa_labels.py)")
    parser.add_argument("--date_run", help="Reuse an existing date_run directory (for patching results)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be submitted without actually submitting")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create or reuse date run identifier
    date_run = args.date_run if args.date_run else datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Create output directory
    output_dir = Path(config['data']['base_output_dir']) / date_run
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config file to output directory for reproducibility
    config_source = Path(args.config)
    config_dest = output_dir / "config.yaml"
    shutil.copy2(config_source, config_dest)
    
    # Create logs directory
    logs_dir = Path("/data/rauschecker2/jkw/aria/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"ARIA Labeling Job Submission")
    print(f"Date run: {date_run}")
    print(f"Output directory: {output_dir}")
    print(f"Config copied to: {config_dest}")
    print(f"Models to process: {len(config['models'])}")
    print()
    
    # Submit jobs for all models
    job_ids = []
    for model in config['models']:
        job_id = submit_job(model, date_run, Path(args.config).resolve(), args.conditions, args.dry_run)
        if job_id:
            job_ids.append(job_id)
    
    if not args.dry_run:
        print(f"\nSubmitted {len(job_ids)} jobs")
        print(f"Job IDs: {', '.join(job_ids)}")
        print(f"\nMonitor with: squeue -u $USER")
        print(f"View logs: ls {logs_dir}/")
        print(f"Results will be saved to: {output_dir}/")
    else:
        print(f"\nDry run completed. Would have submitted {len(config['models'])} jobs.")


if __name__ == "__main__":
    main()
