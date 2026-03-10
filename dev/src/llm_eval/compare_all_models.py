#!/usr/bin/env python3
"""
ARIA Multi-Model Comparison Tool
Compares predictions from multiple LLM models with ground truth labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')
from glob import glob
import json
import re

# Import metrics calculation libraries
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration for column mapping
VARIABLES = [
    'aria-e',
    'aria-h',
    'edema',
    'effusion',
    'microhemorrhage',
    'superficial siderosis'
]

# Excel column names (ground truth) - matching actual column names from the file
EXCEL_COLUMN_MAPPING = {
    'aria-e': 'ARIA-E',
    'aria-h': 'ARIA-H',
    'edema': 'Edema',
    'effusion': 'Effusion',
    'microhemorrhage': 'Microhemorrhage',
    'superficial siderosis': 'Superficial Siderosis'
}

# Model tier classification (for cost/performance analysis)
MODEL_TIERS = {
    'nano': ['gpt_5_nano', 'gpt_35_turbo'],
    'mini': ['gpt_4o_mini', 'gpt_5_mini', 'o1_mini', 'o3_mini', 'o4_mini', 'haiku_4-5'],
    'standard': ['gpt_4_turbo', 'gpt_4o', 'claude_3_5_sonnet'],
    'advanced': ['gpt_4.1', 'gpt_5', 'claude_sonnet_4', 'o1_2024'],
    'premium': ['claude_opus_4-1', 'claude_opus_4-5', 'o1_2024_12']
}

def extract_model_name_from_filename(filename):
    """Extract model name from CSV filename."""
    # Remove 'aria_labels_' prefix and '.csv' suffix
    name = filename.replace('aria_labels_', '').replace('.csv', '')
    # Replace underscores with dots for version numbers
    name = name.replace('_', '.')
    return name


def get_csv_column_for_model(model_name, variable):
    """Get the CSV column name for a specific model and variable."""
    # Map variable names to CSV column format
    var_mapping = {
        'aria-e': 'aria_e',
        'aria-h': 'aria_h',
        'edema': 'edema',
        'effusion': 'effusion',
        'microhemorrhage': 'microhemorrhage',
        'superficial siderosis': 'superficial_siderosis'
    }

    var_csv = var_mapping.get(variable, variable)
    return f"{var_csv}_{model_name}"


def find_prediction_column(pred_columns, variable, model_name):
    """
    Find the best prediction column for a variable in a *single-model* CSV.

    Important: only search within prediction CSV columns (not merged/GT columns),
    otherwise variables like "Superficial Siderosis" can accidentally match the GT
    column and yield artificially perfect metrics.
    """
    var_mapping = {
        'aria-e': 'aria_e',
        'aria-h': 'aria_h',
        'edema': 'edema',
        'effusion': 'effusion',
        'microhemorrhage': 'microhemorrhage',
        'superficial siderosis': 'superficial_siderosis'
    }

    base = var_mapping.get(variable, variable)
    base_norm = re.sub(r'[^a-z0-9]+', '', str(base).lower())

    cols = [str(c) for c in pred_columns if str(c).lower() != 'accession']
    norm_to_col = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in cols}

    model_variants = {
        str(model_name),
        str(model_name).replace('.', '_'),
        str(model_name).replace('.', '-'),
    }

    candidates = []
    candidates.append(base)
    candidates.append(base.replace('_', ' '))
    for mv in sorted(model_variants):
        candidates.append(f"{base}_{mv}")
        candidates.append(f"{base}_{mv}".replace('.', '_'))
        candidates.append(f"{base}_{mv}".replace('.', '-'))

    # 1) Exact match
    for c in candidates:
        if c in cols:
            return c

    # 2) Normalized exact match
    for c in candidates:
        cn = re.sub(r'[^a-z0-9]+', '', str(c).lower())
        if cn in norm_to_col:
            return norm_to_col[cn]

    # 3) Substring match among prediction columns only
    hits = [c for c in cols if base_norm and base_norm in re.sub(r'[^a-z0-9]+', '', c.lower())]
    if len(hits) == 1:
        return hits[0]

    return None


def load_csv_predictions(csv_path):
    """Load prediction CSV file with proper column handling."""
    df = pd.read_csv(csv_path)

    # Rename accession column if needed
    if 'accession number' in df.columns:
        df = df.rename(columns={'accession number': 'accession'})
    elif 'Accession Number' in df.columns:
        df = df.rename(columns={'Accession Number': 'accession'})

    # Convert -1 to 0 (missing/negative)
    df = df.replace(-1, 0)

    # Ensure binary values (0 or 1) for all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'accession':
            df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

    return df


def load_excel_ground_truth(excel_path):
    """Load ground truth Excel file."""
    try:
        df = pd.read_excel(excel_path)
    except:
        df = pd.read_excel(excel_path, engine='openpyxl')

    # Standardize accession column name
    if 'Accession' in df.columns:
        df = df.rename(columns={'Accession': 'accession'})

    # Convert -1 to 0 (missing/negative)
    df = df.replace(-1, 0)

    # Ensure binary values for all variables
    for var in VARIABLES:
        excel_col = EXCEL_COLUMN_MAPPING.get(var)
        if excel_col in df.columns:
            df[excel_col] = df[excel_col].apply(lambda x: 1 if x > 0 else 0)

    return df


def calculate_binary_metrics(y_true, y_pred):
    """Calculate comprehensive binary classification metrics."""
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return None

    # Calculate confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except:
        return None

    # Calculate metrics
    metrics = {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Total': len(y_true),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    }

    # Try to calculate AUC
    try:
        if len(np.unique(y_true)) > 1:
            metrics['AUC'] = roc_auc_score(y_true, y_pred)
        else:
            metrics['AUC'] = np.nan
    except:
        metrics['AUC'] = np.nan

    return metrics


def process_all_models(csv_dir, excel_gt_path, dir_exp):
    """Process all CSV files and calculate metrics for each model."""
    # Get all CSV files
    csv_files = sorted(Path(csv_dir).glob('*.csv'))
    print(f"\nFound {len(csv_files)} model prediction files")

    # Load ground truth
    df_gt = load_excel_ground_truth(excel_gt_path)

    # Store all results
    all_results = {}

    # Process each CSV file
    for csv_file in csv_files:
        model_name = extract_model_name_from_filename(csv_file.name)
        print(f"\nProcessing model: {model_name}")

        # Load predictions
        df_pred = load_csv_predictions(csv_file)

        # Merge with ground truth
        df_pred['accession'] = df_pred['accession'].astype(str)
        df_gt['accession'] = df_gt['accession'].astype(str)
        merged = pd.merge(df_pred, df_gt, on='accession', suffixes=('_pred', '_gt'))

        if len(merged) == 0:
            print(f"  Warning: No matching accessions for {model_name}")
            continue

        # Calculate metrics for each variable
        model_results = {}
        for var in VARIABLES:
            # Find the prediction column from the *prediction CSV* (avoid matching GT columns)
            pred_col = find_prediction_column(df_pred.columns, var, model_name)
            if pred_col is None:
                continue

            gt_col = EXCEL_COLUMN_MAPPING[var]

            if gt_col not in merged.columns:
                continue

            # Handle overlapping column names from merge suffixing.
            if pred_col not in merged.columns:
                pred_col_suffixed = f"{pred_col}_pred"
                if pred_col_suffixed in merged.columns:
                    pred_col = pred_col_suffixed
                else:
                    continue

            # Extract data
            y_pred = merged[pred_col].values
            y_true = merged[gt_col].values

            # Remove NaN values
            mask = ~(pd.isna(y_pred) | pd.isna(y_true))
            y_pred = y_pred[mask]
            y_true = y_true[mask]

            # Calculate metrics
            metrics = calculate_binary_metrics(y_true, y_pred)
            if metrics:
                model_results[var] = metrics

        all_results[model_name] = model_results

    return all_results


def aggregate_results(all_results, dir_exp):
    """Aggregate results across all models and create summary tables."""
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)

    # Create summary DataFrame
    summary_rows = []

    for model_name, model_results in all_results.items():
        for var in VARIABLES:
            if var in model_results:
                metrics = model_results[var]
                row = {
                    'Model': model_name,
                    'Variable': var,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1': metrics['F1'],
                    'Sensitivity': metrics['Sensitivity'],
                    'Specificity': metrics['Specificity'],
                    'AUC': metrics.get('AUC', np.nan)
                }
                summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # Save full summary
    summary_path = dir_exp / 'all_models_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\nFull summary saved to: {summary_path}")

    # Create average performance by model
    model_avg = df_summary.groupby('Model').agg({
        'Accuracy': 'mean',
        'F1': 'mean',
        'AUC': 'mean',
        'Sensitivity': 'mean',
        'Specificity': 'mean'
    }).round(4)

    model_avg['Overall_Score'] = (
        model_avg['Accuracy'] + model_avg['F1'] +
        model_avg['AUC'].fillna(model_avg['AUC'].mean())
    ) / 3

    model_avg = model_avg.sort_values('Overall_Score', ascending=False)

    # Save model ranking
    ranking_path = dir_exp / 'model_ranking.csv'
    model_avg.to_csv(ranking_path)
    print(f"Model ranking saved to: {ranking_path}")

    return df_summary, model_avg


def create_visualizations(df_summary, model_avg, dir_exp):
    """Create comparison visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # 1. Model ranking bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    top_10 = model_avg.head(10)
    bars = ax.barh(range(len(top_10)), top_10['Overall_Score'])

    # Color bars by performance
    colors = plt.cm.RdYlGn(top_10['Overall_Score'] / top_10['Overall_Score'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index)
    ax.set_xlabel('Overall Score', fontsize=12)
    ax.set_title('Top 10 Models by Overall Performance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    ranking_plot = dir_exp / 'model_ranking_chart.png'
    plt.savefig(ranking_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model ranking chart saved to: {ranking_plot}")

    # 2. Create three heatmaps for different metrics
    metrics_to_plot = [
        ('F1', 'F1 Score', 'RdYlGn'),
        ('Accuracy', 'Accuracy', 'RdYlGn'),
        ('AUC', 'AUC Score', 'RdYlGn')
    ]

    for metric_name, metric_label, cmap in metrics_to_plot:
        pivot_data = df_summary.pivot(index='Model', columns='Variable', values=metric_name)

        plt.figure(figsize=(12, len(pivot_data) * 0.5))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap,
                    vmin=0, vmax=1, cbar_kws={'label': metric_label})
        plt.title(f'Model Performance Heatmap ({metric_label})', fontsize=14, fontweight='bold')
        plt.xlabel('Variable', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()

        heatmap_path = dir_exp / f'heatmap_{metric_name.lower()}.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{metric_label} heatmap saved to: {heatmap_path}")

    # 3. Create combined comparison heatmap (all metrics side by side)
    fig, axes = plt.subplots(1, 3, figsize=(20, len(df_summary['Model'].unique()) * 0.5))

    for idx, (metric_name, metric_label, cmap) in enumerate(metrics_to_plot):
        pivot_data = df_summary.pivot(index='Model', columns='Variable', values=metric_name)
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap,
                    vmin=0, vmax=1, cbar_kws={'label': metric_label},
                    ax=axes[idx])
        axes[idx].set_title(f'{metric_label}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Variable' if idx == 1 else '', fontsize=10)
        axes[idx].set_ylabel('Model' if idx == 0 else '', fontsize=10)

    plt.suptitle('Model Performance Comparison - All Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()

    combined_path = dir_exp / 'heatmap_combined_metrics.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined metrics heatmap saved to: {combined_path}")


def find_best_models(model_avg, dir_exp):
    """Identify best cheap and intensive models."""
    print("\n" + "="*60)
    print("BEST MODEL RECOMMENDATIONS")
    print("="*60)

    # Define cheap and intensive models based on naming patterns
    cheap_models = [m for m in model_avg.index if any(x in m.lower() for x in ['mini', 'nano', 'turbo', '35'])]
    intensive_models = [m for m in model_avg.index if any(x in m.lower() for x in ['opus', 'o1.2024', 'gpt.5', 'claude.4'])]

    recommendations = {}

    if cheap_models:
        best_cheap = model_avg.loc[cheap_models].sort_values('Overall_Score', ascending=False).iloc[0]
        recommendations['best_cheap'] = best_cheap.name
        print(f"\nBest Cheap Model: {best_cheap.name}")
        print(f"  Overall Score: {best_cheap['Overall_Score']:.4f}")
        print(f"  Accuracy: {best_cheap['Accuracy']:.4f}")
        print(f"  F1: {best_cheap['F1']:.4f}")

    if intensive_models:
        best_intensive = model_avg.loc[intensive_models].sort_values('Overall_Score', ascending=False).iloc[0]
        recommendations['best_intensive'] = best_intensive.name
        print(f"\nBest Intensive Model: {best_intensive.name}")
        print(f"  Overall Score: {best_intensive['Overall_Score']:.4f}")
        print(f"  Accuracy: {best_intensive['Accuracy']:.4f}")
        print(f"  F1: {best_intensive['F1']:.4f}")

    # Overall best
    best_overall = model_avg.iloc[0]
    recommendations['best_overall'] = best_overall.name
    print(f"\nBest Overall Model: {best_overall.name}")
    print(f"  Overall Score: {best_overall['Overall_Score']:.4f}")
    print(f"  Accuracy: {best_overall['Accuracy']:.4f}")
    print(f"  F1: {best_overall['F1']:.4f}")

    # Save recommendations
    rec_path = dir_exp / 'model_recommendations.json'
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"\nRecommendations saved to: {rec_path}")

    return recommendations


def main():
    """Main execution function."""
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Compare all model prediction CSVs against ground truth.")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=repo_root / "data/ucsf_aria/labeled-llm",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=repo_root / "data/ucsf_aria/labeled/combined_annotations.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / f"exp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_multi_model_comparison",
    )
    args = parser.parse_args()

    csv_dir = args.predictions_dir
    excel_gt_path = args.ground_truth
    dir_exp = args.output_dir
    dir_exp.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ARIA Multi-Model Comparison Tool")
    print("="*60)
    print(f"Output directory: {dir_exp}")

    # Process all models
    all_results = process_all_models(csv_dir, excel_gt_path, dir_exp)

    if not all_results:
        print("\nERROR: No results to process!")
        return

    # Aggregate results
    df_summary, model_avg = aggregate_results(all_results, dir_exp)

    # Create visualizations
    create_visualizations(df_summary, model_avg, dir_exp)

    # Find best models
    recommendations = find_best_models(model_avg, dir_exp)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {dir_exp}")
    print("\nGenerated files:")
    print("  - all_models_summary.csv        : Detailed metrics for all models")
    print("  - model_ranking.csv             : Models ranked by performance")
    print("  - model_ranking_chart.png       : Visual ranking of top models")
    print("  - heatmap_f1.png                : Heatmap of F1 scores")
    print("  - heatmap_accuracy.png          : Heatmap of accuracy scores")
    print("  - heatmap_auc.png               : Heatmap of AUC scores")
    print("  - heatmap_combined_metrics.png  : Side-by-side comparison of all metrics")
    print("  - model_recommendations.json    : Best model recommendations")


if __name__ == "__main__":
    main()
