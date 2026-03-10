#!/usr/bin/env python3
"""
ARIA Prediction Comparison Tool
Compares model predictions with ground truth labels and generates comprehensive metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

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

# Column name mapping from CSV format to standard names
# CSV_COLUMN_MAPPING = {
#     'aria-e': 'aria_e_o1-mini-2024-09-12',
#     'aria-h': 'aria_h_o1-mini-2024-09-12',
#     'edema': 'edema_o1-mini-2024-09-12',
#     'effusion': 'effusion_o1-mini-2024-09-12',
#     'microhemorrhage': 'microhemorrhage_o1-mini-2024-09-12',
#     'superficial siderosis': 'superficial_siderosis_o1-mini-2024-09-12'
# }

# CSV_COLUMN_MAPPING = {
#     'aria-e': 'aria_e_us.anthropic.claude-opus-4-1-20250805-v1:0',
#     'aria-h': 'aria_h_us.anthropic.claude-opus-4-1-20250805-v1:0',
#     'edema': 'edema_us.anthropic.claude-opus-4-1-20250805-v1:0',
#     'effusion': 'effusion_us.anthropic.claude-opus-4-1-20250805-v1:0',
#     'microhemorrhage': 'microhemorrhage_us.anthropic.claude-opus-4-1-20250805-v1:0',
#     'superficial siderosis': 'superficial_siderosis_us.anthropic.claude-opus-4-1-20250805-v1:0'
# }

CSV_COLUMN_MAPPING = {
    'aria-e': 'aria_e_gpt-5-2025-08-07',
    'aria-h': 'aria_h_gpt-5-2025-08-07',
    'edema': 'edema_gpt-5-2025-08-07',
    'effusion': 'effusion_gpt-5-2025-08-07',
    'microhemorrhage': 'microhemorrhage_gpt-5-2025-08-07',
    'superficial siderosis': 'superficial_siderosis_gpt-5-2025-08-07'
}

# Excel column names (ground truth) - matching actual column names from the file
EXCEL_COLUMN_MAPPING = {
    'aria-e': 'ARIA-E',
    'aria-h': 'ARIA-H',
    'edema': 'Edema',
    'effusion': 'Effusion',
    'microhemorrhage': 'Microhemorrhage',
    'superficial siderosis': 'Superficial Siderosis'
}


def load_csv_predictions(csv_path):
    """
    Load prediction CSV file with proper column handling.

    Args:
        csv_path: Path to CSV file with predictions

    Returns:
        DataFrame with standardized column names
    """
    print(f"Loading CSV predictions from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Print available columns for debugging
    print(f"Available columns in CSV: {list(df.columns)[:10]}...")  # Show first 10 columns

    # Rename accession column if needed
    if 'accession number' in df.columns:
        df = df.rename(columns={'accession number': 'accession'})
    elif 'Accession Number' in df.columns:
        df = df.rename(columns={'Accession Number': 'accession'})

    # Convert -1 to 0 (missing/negative)
    df = df.replace(-1, 0)

    # Ensure binary values (0 or 1)
    for var in VARIABLES:
        csv_col = CSV_COLUMN_MAPPING.get(var)
        if csv_col in df.columns:
            df[csv_col] = df[csv_col].apply(lambda x: 1 if x > 0 else 0)

    print(f"Loaded {len(df)} prediction records")
    return df


def load_excel_ground_truth(excel_path):
    """
    Load ground truth Excel file.

    Args:
        excel_path: Path to Excel file with ground truth

    Returns:
        DataFrame with ground truth data
    """
    print(f"Loading Excel ground truth from: {excel_path}")

    # Try to read Excel file
    try:
        df = pd.read_excel(excel_path)
    except:
        # If it fails, try with engine='openpyxl'
        df = pd.read_excel(excel_path, engine='openpyxl')

    # Print available columns for debugging
    print(f"Available columns in Excel: {list(df.columns)[:10]}...")  # Show first 10 columns

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

    print(f"Loaded {len(df)} ground truth records")
    return df


def merge_datasets(df_pred, df_gt):
    """
    Merge prediction and ground truth datasets on accession number.

    Args:
        df_pred: DataFrame with predictions
        df_gt: DataFrame with ground truth

    Returns:
        Merged DataFrame with both predictions and ground truth
    """
    print("\nMerging datasets on accession number...")

    # Ensure accession columns are string type for proper matching
    df_pred['accession'] = df_pred['accession'].astype(str)
    df_gt['accession'] = df_gt['accession'].astype(str)

    # Merge on accession
    merged = pd.merge(df_pred, df_gt, on='accession', suffixes=('_pred', '_gt'))

    print(f"Found {len(merged)} matching accession numbers")
    print(f"Predictions without match: {len(df_pred) - len(merged)}")
    print(f"Ground truth without match: {len(df_gt) - len(merged)}")

    return merged


def calculate_binary_metrics(y_true, y_pred):
    """
    Calculate comprehensive binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dictionary with all binary classification metrics
    """
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    metrics = {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Total': len(y_true),

        # Basic metrics
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),  # Same as Sensitivity
        'F1': f1_score(y_true, y_pred, zero_division=0),

        # Additional metrics
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),  # TPR
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # TNR
        'TPR': recall_score(y_true, y_pred, zero_division=0),  # True Positive Rate
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
        'PPV': precision_score(y_true, y_pred, zero_division=0),  # Positive Predictive Value
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
    }

    # Try to calculate AUC if we have both classes
    try:
        if len(np.unique(y_true)) > 1:
            metrics['AUC'] = roc_auc_score(y_true, y_pred)
        else:
            metrics['AUC'] = np.nan
    except:
        metrics['AUC'] = np.nan

    return metrics


def calculate_all_metrics(merged_df, dir_exp):
    """
    Calculate metrics for all variables.

    Args:
        merged_df: Merged DataFrame with predictions and ground truth
        dir_exp: Directory to save results

    Returns:
        Dictionary with metrics for each variable
    """
    print("\n" + "="*60)
    print("CALCULATING BINARY CLASSIFICATION METRICS")
    print("="*60)

    all_metrics = {}

    for var in VARIABLES:
        print(f"\nProcessing: {var}")
        print("-" * 40)

        # Get column names
        csv_col = CSV_COLUMN_MAPPING.get(var)
        excel_col = EXCEL_COLUMN_MAPPING.get(var)

        # Check if columns exist in merged data
        if csv_col not in merged_df.columns:
            print(f"WARNING: Prediction column '{csv_col}' not found in data")
            continue

        if excel_col not in merged_df.columns:
            print(f"WARNING: Ground truth column '{excel_col}' not found in data")
            continue

        # Extract predictions and ground truth
        y_pred = merged_df[csv_col].values
        y_true = merged_df[excel_col].values

        # Remove NaN values
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        print(f"Valid samples: {len(y_pred)}")

        if len(y_pred) == 0:
            print(f"No valid samples for {var}")
            continue

        # Calculate metrics
        metrics = calculate_binary_metrics(y_true, y_pred)
        all_metrics[var] = metrics

        # Print summary
        print(f"Accuracy: {metrics['Accuracy']:.3f}")
        print(f"Sensitivity: {metrics['Sensitivity']:.3f}")
        print(f"Specificity: {metrics['Specificity']:.3f}")
        print(f"F1 Score: {metrics['F1']:.3f}")
        print(f"AUC: {metrics['AUC']:.3f}" if not np.isnan(metrics['AUC']) else "AUC: N/A")

    return all_metrics


def plot_roc_curves(merged_df, all_metrics, dir_exp):
    """
    Generate ROC curves for all variables.

    Args:
        merged_df: Merged DataFrame with predictions and ground truth
        all_metrics: Dictionary with calculated metrics
        dir_exp: Directory to save plots

    Returns:
        None (saves plots to disk)
    """
    print("\n" + "="*60)
    print("GENERATING ROC CURVES")
    print("="*60)

    # Create figure with subplots
    n_vars = len(VARIABLES)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    plot_idx = 0

    for var in VARIABLES:
        if var not in all_metrics:
            print(f"Skipping ROC for {var} - no metrics available")
            continue

        print(f"Plotting ROC curve for: {var}")

        # Get column names
        csv_col = CSV_COLUMN_MAPPING.get(var)
        excel_col = EXCEL_COLUMN_MAPPING.get(var)

        if csv_col not in merged_df.columns or excel_col not in merged_df.columns:
            continue

        # Extract data
        y_pred = merged_df[csv_col].values
        y_true = merged_df[excel_col].values

        # Remove NaN values
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        # Calculate ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
        except:
            print(f"Could not calculate ROC for {var}")
            plot_idx += 1
            continue

        # Plot on subplot
        ax = axes[plot_idx]
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'ROC Curve: {var}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save figure
    roc_path = dir_exp / 'roc_curves.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curves saved to: {roc_path}")

    plt.close()

    # Also create individual ROC plots for better visibility
    print("\nGenerating individual ROC plots...")

    for var in VARIABLES:
        if var not in all_metrics:
            continue

        # Get column names
        csv_col = CSV_COLUMN_MAPPING.get(var)
        excel_col = EXCEL_COLUMN_MAPPING.get(var)

        if csv_col not in merged_df.columns or excel_col not in merged_df.columns:
            continue

        # Extract data
        y_pred = merged_df[csv_col].values
        y_true = merged_df[excel_col].values

        # Remove NaN values
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
        except:
            continue

        # Create individual plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve: {var}', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add AUC text
        plt.text(0.6, 0.2, f'AUC = {auc:.3f}', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save individual plot
        individual_path = dir_exp / f'roc_{var.replace(" ", "_")}.png'
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  - Saved: {individual_path}")


def plot_confusion_matrices(merged_df, all_metrics, dir_exp):
    """
    Generate confusion matrix heatmaps for all variables.

    Args:
        merged_df: Merged DataFrame with predictions and ground truth
        all_metrics: Dictionary with calculated metrics
        dir_exp: Directory to save plots

    Returns:
        None (saves plots to disk)
    """
    print("\n" + "="*60)
    print("GENERATING CONFUSION MATRICES")
    print("="*60)

    # Create figure with subplots
    n_vars = len(VARIABLES)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    plot_idx = 0

    for var in VARIABLES:
        if var not in all_metrics:
            print(f"Skipping confusion matrix for {var} - no metrics available")
            continue

        print(f"Plotting confusion matrix for: {var}")

        # Get column names
        csv_col = CSV_COLUMN_MAPPING.get(var)
        excel_col = EXCEL_COLUMN_MAPPING.get(var)

        if csv_col not in merged_df.columns or excel_col not in merged_df.columns:
            continue

        # Extract data
        y_pred = merged_df[csv_col].values
        y_true = merged_df[excel_col].values

        # Remove NaN values
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot on subplot
        ax = axes[plot_idx]

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    square=True, cbar=True, ax=ax,
                    annot_kws={'size': 14, 'weight': 'bold'})

        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'Confusion Matrix: {var}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
        ax.set_yticklabels(['Negative (0)', 'Positive (1)'])

        # Add metrics text below matrix
        metrics = all_metrics[var]
        text_str = f"Acc: {metrics['Accuracy']:.3f} | Sens: {metrics['Sensitivity']:.3f} | Spec: {metrics['Specificity']:.3f}"
        ax.text(0.5, -0.15, text_str, transform=ax.transAxes,
                ha='center', fontsize=9, style='italic')

        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save figure
    cm_path = dir_exp / 'confusion_matrices.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrices saved to: {cm_path}")

    plt.close()

    # Also create individual confusion matrix plots for better visibility
    print("\nGenerating individual confusion matrix plots...")

    for var in VARIABLES:
        if var not in all_metrics:
            continue

        # Get column names
        csv_col = CSV_COLUMN_MAPPING.get(var)
        excel_col = EXCEL_COLUMN_MAPPING.get(var)

        if csv_col not in merged_df.columns or excel_col not in merged_df.columns:
            continue

        # Extract data
        y_pred = merged_df[csv_col].values
        y_true = merged_df[excel_col].values

        # Remove NaN values
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Get metrics
        metrics = all_metrics[var]

        # Create individual plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create annotated heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    square=True, cbar=True, ax=ax,
                    annot_kws={'size': 20, 'weight': 'bold'},
                    cbar_kws={'label': 'Count'})

        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_title(f'Confusion Matrix: {var}', fontsize=16, fontweight='bold')
        ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)
        ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)

        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum() * 100
        for i in range(2):
            for j in range(2):
                text = ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                             ha='center', va='center', color='darkblue', fontsize=11)

        # Add metrics box
        metrics_text = (f"Accuracy: {metrics['Accuracy']:.3f}\n"
                       f"Sensitivity: {metrics['Sensitivity']:.3f}\n"
                       f"Specificity: {metrics['Specificity']:.3f}\n"
                       f"Precision: {metrics['Precision']:.3f}\n"
                       f"F1 Score: {metrics['F1']:.3f}")

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', bbox=props)

        plt.tight_layout()

        # Save individual plot
        individual_path = dir_exp / f'confusion_matrix_{var.replace(" ", "_")}.png'
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  - Saved: {individual_path}")


def export_metrics_to_csv(all_metrics, dir_exp):
    """
    Export all metrics to a CSV file.

    Args:
        all_metrics: Dictionary with calculated metrics for each variable
        dir_exp: Directory to save the CSV file

    Returns:
        None (saves CSV to disk)
    """
    print("\n" + "="*60)
    print("EXPORTING METRICS TO CSV")
    print("="*60)

    # Create a list to store all rows
    rows = []

    # Define the order of metrics for the CSV
    metric_order = [
        'TP', 'TN', 'FP', 'FN', 'Total',
        'TPR', 'FPR',
        'Sensitivity', 'Specificity',
        'F1', 'Recall', 'Accuracy', 'Precision',
        'NPV', 'PPV', 'AUC'
    ]

    for var in VARIABLES:
        if var not in all_metrics:
            print(f"No metrics for {var} - skipping")
            continue

        metrics = all_metrics[var]

        # Create row with variable name first
        row = {'Variable': var}

        # Add all metrics in specified order
        for metric in metric_order:
            if metric in metrics:
                value = metrics[metric]
                # Format float values to 4 decimal places
                if isinstance(value, float):
                    row[metric] = round(value, 4)
                else:
                    row[metric] = value
            else:
                row[metric] = 'N/A'

        rows.append(row)

    # Create DataFrame
    df_metrics = pd.DataFrame(rows)

    # Check if we have any metrics to export
    if df_metrics.empty:
        print("No metrics to export - DataFrame is empty")
        # Create an empty CSV with headers
        csv_path = dir_exp / 'metrics_summary.csv'
        with open(csv_path, 'w') as f:
            f.write('Variable,' + ','.join(metric_order) + '\n')
        print(f"Empty metrics file created at: {csv_path}")
    else:
        # Save to CSV
        csv_path = dir_exp / 'metrics_summary.csv'
        df_metrics.to_csv(csv_path, index=False)
        print(f"Metrics exported to: {csv_path}")

    # Also create a detailed metrics report
    print("\nCreating detailed metrics report...")

    # Create detailed report with formatted output
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ARIA PREDICTION COMPARISON - DETAILED METRICS REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    for var in VARIABLES:
        if var not in all_metrics:
            continue

        metrics = all_metrics[var]
        report_lines.append("-"*80)
        report_lines.append(f"VARIABLE: {var.upper()}")
        report_lines.append("-"*80)

        # Confusion Matrix
        report_lines.append("\nConfusion Matrix:")
        report_lines.append(f"  True Positives (TP):  {metrics['TP']:6d}")
        report_lines.append(f"  True Negatives (TN):  {metrics['TN']:6d}")
        report_lines.append(f"  False Positives (FP): {metrics['FP']:6d}")
        report_lines.append(f"  False Negatives (FN): {metrics['FN']:6d}")
        report_lines.append(f"  Total Samples:        {metrics['Total']:6d}")

        # Performance Metrics
        report_lines.append("\nPerformance Metrics:")
        report_lines.append(f"  Accuracy:             {metrics['Accuracy']:.4f}")
        report_lines.append(f"  Precision (PPV):      {metrics['Precision']:.4f}")
        report_lines.append(f"  Recall (Sensitivity): {metrics['Recall']:.4f}")
        report_lines.append(f"  Specificity:          {metrics['Specificity']:.4f}")
        report_lines.append(f"  F1 Score:             {metrics['F1']:.4f}")

        # Additional Metrics
        report_lines.append("\nAdditional Metrics:")
        report_lines.append(f"  True Positive Rate:   {metrics['TPR']:.4f}")
        report_lines.append(f"  False Positive Rate:  {metrics['FPR']:.4f}")
        report_lines.append(f"  Negative Pred Value:  {metrics['NPV']:.4f}")

        if not pd.isna(metrics.get('AUC', np.nan)):
            report_lines.append(f"  AUC Score:            {metrics['AUC']:.4f}")
        else:
            report_lines.append(f"  AUC Score:            N/A")

        report_lines.append("")

    # Add summary statistics
    report_lines.append("="*80)
    report_lines.append("SUMMARY STATISTICS ACROSS ALL VARIABLES")
    report_lines.append("="*80)

    # Calculate average metrics
    avg_metrics = {}
    for metric in metric_order:
        values = []
        for var in VARIABLES:
            if var in all_metrics and metric in all_metrics[var]:
                value = all_metrics[var][metric]
                if isinstance(value, (int, float)) and not pd.isna(value):
                    values.append(value)

        if values and metric not in ['TP', 'TN', 'FP', 'FN', 'Total']:
            avg_metrics[metric] = np.mean(values)

    report_lines.append("\nAverage Performance Across Variables:")
    for metric, value in avg_metrics.items():
        report_lines.append(f"  {metric:20s}: {value:.4f}")

    # Save detailed report
    report_path = dir_exp / 'metrics_detailed_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Detailed report saved to: {report_path}")

    # Create a transposed version for easier reading (only if we have data)
    if not df_metrics.empty:
        print("\nCreating transposed metrics CSV...")

        # Transpose the metrics DataFrame
        df_transposed = df_metrics.set_index('Variable').T
        df_transposed.index.name = 'Metric'

        # Save transposed version
        transposed_path = dir_exp / 'metrics_summary_transposed.csv'
        df_transposed.to_csv(transposed_path)
        print(f"Transposed metrics saved to: {transposed_path}")
    else:
        print("\nSkipping transposed CSV creation - no metrics available")

    return df_metrics


def main():
    """
    Main execution function - loads data, calculates metrics.
    """
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Detailed comparison for one prediction CSV.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=repo_root / "data/ucsf_aria/labeled-llm/aria_labels_gpt_5_2025_08_07.csv",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=repo_root / "data/ucsf_aria/labeled/combined_annotations.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / f"exp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_comparison_results",
    )
    args = parser.parse_args()

    csv_pred_path = args.predictions
    excel_gt_path = args.ground_truth
    dir_exp = args.output_dir
    dir_exp.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ARIA Prediction Comparison Tool")
    print("="*60)

    # Load data
    df_pred = load_csv_predictions(csv_pred_path)
    df_gt = load_excel_ground_truth(excel_gt_path)

    # Merge datasets
    merged_df = merge_datasets(df_pred, df_gt)

    if len(merged_df) == 0:
        print("\nERROR: No matching accession numbers found!")
        print("Please check that accession numbers match between files.")
        return

    print("\nData loading complete!")
    print(f"Ready to analyze {len(merged_df)} matched records")

    # Save merged data for inspection
    merged_path = dir_exp / 'merged_data.csv'
    merged_df.to_csv(merged_path, index=False)
    print(f"\nMerged data saved to: {merged_path}")

    # Calculate metrics for all variables
    all_metrics = calculate_all_metrics(merged_df, dir_exp)

    print("\n" + "="*60)
    print("METRICS CALCULATION COMPLETE")
    print("="*60)
    print(f"Processed {len(all_metrics)} variables successfully")

    # Generate ROC curves
    plot_roc_curves(merged_df, all_metrics, dir_exp)

    # Generate confusion matrices
    plot_confusion_matrices(merged_df, all_metrics, dir_exp)

    # Export metrics to CSV
    metrics_df = export_metrics_to_csv(all_metrics, dir_exp)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {dir_exp}")
    print("\nGenerated files:")
    print("  - merged_data.csv           : Combined prediction and ground truth data")
    print("  - metrics_summary.csv       : All metrics in tabular format")
    print("  - metrics_summary_transposed.csv : Transposed metrics for easier reading")
    print("  - metrics_detailed_report.txt    : Detailed text report with all metrics")
    print("  - roc_curves.png            : Combined ROC curves for all variables")
    print("  - roc_*.png                 : Individual ROC curves for each variable")
    print("  - confusion_matrices.png    : Combined confusion matrices")
    print("  - confusion_matrix_*.png    : Individual confusion matrices")


if __name__ == "__main__":
    main()
