# aria-prediction
Predicting ARIA through a combination of imaging and clinical variables.

aria-prediction/
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   ├── config.yaml.template
│   └── example_config.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data_processing/
│   ├── models/
│   └── utils/
├── exp/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
└── scripts/
    └── run_pipeline.sh

# ARIA Prediction Pipeline

A configurable machine learning pipeline for ARIA prediction analysis.

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd aria-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the pipeline**
   ```bash
   cp config.yaml.template config.yaml
   # Edit config.yaml with your specific settings
   ```

## Configuration

The pipeline uses a YAML configuration file (`config.yaml`) to specify:
- **Data sources**: Path to your input data
- **Output settings**: Where results should be saved (within `exp/` directory)
- **Model hyperparameters**: All model-specific settings
- **Pipeline settings**: Preprocessing steps, evaluation metrics

### Example Configuration

```yaml
data:
  source_path: "data/raw/aria_data.csv"
  processed_data_dir: "data/processed"

output:
  base_dir: "exp"
  experiment_name: "aria_rf_experiment"

model:
  type: "random_forest"
  parameters:
    n_estimators: 100
    max_depth: 10
```

## Usage

### Basic Usage
```bash
python main.py
```

### With Custom Config
```bash
python main.py --config my_custom_config.yaml
```

## Output Structure

Each run creates a timestamped directory in `exp/` containing:
```
exp/experiment_20231201_143022/
├── config.yaml          # Copy of the configuration used
├── models/              # Trained model files
├── predictions/         # Model predictions
├── plots/              # Visualization outputs
└── logs/               # Detailed execution logs
```

## Privacy and Security

- **Private data and results are automatically excluded from Git**
- The `.gitignore` file ensures that:
  - All experiment outputs (`exp/*/`) stay local
  - Private data files (`data/*/`) are not committed
  - Your actual `config.yaml` is not shared (only the template)

## Development

### Project Structure
```
aria-prediction/
├── main.py                   # Main pipeline script
├── config.yaml.template     # Configuration template
├── src/                     # Source code modules
├── exp/                     # Experiment outputs (git-ignored)
├── data/                    # Data files (git-ignored)
└── logs/                    # Log files (git-ignored)
```

### Adding New Features
1. Add your modules to the `src/` directory
2. Update the pipeline logic in `main.py`
3. Add any new dependencies to `requirements.txt`

## Troubleshooting

### Common Issues

1. **Config file not found**
   - Make sure you've copied `config.yaml.template` to `config.yaml`
   - Check that the file is in the same directory as `main.py`

2. **Permission errors on output directory**
   - Ensure you have write permissions to the `exp/` directory
   - Check disk space availability

3. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility