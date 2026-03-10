#!/usr/bin/env python3
"""
ARIA Prediction Pipeline
Main script that parses config.yaml and runs the prediction pipeline.
"""

import os
import yaml
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please copy config.yaml.template to config.yaml and update it with your settings."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_output_directory(config: Dict[str, Any], config_path: str) -> str:
    """
    Create output directory and copy config file.
    Returns the path to the created output directory.
    """
    # Generate experiment name with timestamp if not provided
    experiment_name = config['output'].get('experiment_name')
    if 'YYYYMMDD_HHMMSS' in experiment_name or not experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Create output directory
    base_dir = config['output']['base_dir']
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'predictions', 'plots', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Copy config file to output directory
    config_dest = os.path.join(output_dir, 'config.yaml')
    shutil.copy2(config_path, config_dest)
    
    return output_dir

def setup_logging(config: Dict[str, Any], output_dir: str):
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['level'].upper())
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if enabled
    if config['logging']['log_to_file']:
        log_file = os.path.join(output_dir, 'logs', 'pipeline.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def run_pipeline(config: Dict[str, Any], output_dir: str):
    """
    Main pipeline logic goes here.
    This is where you'll implement your actual ARIA prediction pipeline.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ARIA prediction pipeline")
    logger.info(f"Output directory: {output_dir}")
    
    # TODO: Implement your pipeline steps here
    # 1. Data loading
    logger.info("Loading data...")
    # load_data(config['data'])
    
    # 2. Data preprocessing
    logger.info("Preprocessing data...")
    # preprocess_data(config['pipeline'])
    
    # 3. Model training
    logger.info("Training model...")
    # train_model(config['model'], config['training'])
    
    # 4. Model evaluation
    logger.info("Evaluating model...")
    # evaluate_model(config['pipeline']['evaluation_metrics'])
    
    # 5. Save results
    logger.info("Saving results...")
    # save_results(output_dir)
    
    logger.info("Pipeline completed successfully!")

def main():
    """Main entry point."""
    try:
        # Load configuration
        config = load_config()
        
        # Setup output directory
        output_dir = setup_output_directory(config, "config.yaml")
        
        # Setup logging
        setup_logging(config, output_dir)
        
        # Run the pipeline
        run_pipeline(config, output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())