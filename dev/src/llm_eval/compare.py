from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_predictions(csv_pred):
    """Load prediction CSV file"""
    df_pred = pd.read_csv(csv_pred)
    # Convert -1 to 0 (missing)
    df_pred = df_pred.replace(-1, 0)
    return df_pred

def load_ground_truth(dir_gt):
    """Load all ground truth CSV files and combine them"""
    gt_files = list(dir_gt.glob('*.csv'))
    if not gt_files:
        print(f"No CSV files found in {dir_gt}")
        return None
    
    dfs = []
    for file in gt_files:
        try:
            df = pd.read_csv(file)
            # Convert -1 to 0 (missing)
            df = df.replace(-1, 0)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    return None

def compare_accuracy(df_pred, df_gt):
    """Compare predictions with ground truth and compute statistics"""
    
    # Merge on Accession Number
    if 'Accession Number' in df_pred.columns and 'Accession Number' in df_gt.columns:
        merged = pd.merge(df_pred, df_gt, on='Accession Number', suffixes=('_pred', '_gt'))
    else:
        print("Accession Number column not found in one or both datasets")
        return None
    
    print(f"Found {len(merged)} matching accession numbers")
    
    results = {}
    
    # Check for aria-e and aria-h columns
    aria_e_cols = [col for col in merged.columns if 'aria_e' in col.lower() or 'aria-e' in col.lower()]
    aria_h_cols = [col for col in merged.columns if 'aria_h' in col.lower() or 'aria-h' in col.lower()]
    
    print(f"ARIA-E columns found: {aria_e_cols}")
    print(f"ARIA-H columns found: {aria_h_cols}")
    
    # Compare ARIA-E
    if len(aria_e_cols) >= 2:
        pred_col = [col for col in aria_e_cols if 'pred' in col or any(x in col for x in ['o1', 'mini'])]
        gt_col = [col for col in aria_e_cols if 'gt' in col or col not in pred_col]
        
        if pred_col and gt_col:
            pred_col, gt_col = pred_col[0], gt_col[0]
            y_pred = merged[pred_col]
            y_true = merged[gt_col]
            
            results['aria_e'] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }
    
    # Compare ARIA-H
    if len(aria_h_cols) >= 2:
        pred_col = [col for col in aria_h_cols if 'pred' in col or any(x in col for x in ['o1', 'mini'])]
        gt_col = [col for col in aria_h_cols if 'gt' in col or col not in pred_col]
        
        if pred_col and gt_col:
            pred_col, gt_col = pred_col[0], gt_col[0]
            y_pred = merged[pred_col]
            y_true = merged[gt_col]
            
            results['aria_h'] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }
    
    return results, merged

def print_results(results):
    """Print comparison results"""
    for label, metrics in results.items():
        print(f"\n{label.upper()} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Confusion Matrix:")
        print(metrics['confusion_matrix'])

def parse_args():
    repo_root = Path(__file__).resolve().parents[3]
    return argparse.ArgumentParser(description="Compare one prediction CSV against UCSF ground truth.").parse_args()

def main():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Compare one prediction CSV against UCSF ground truth.")
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
    args = parser.parse_args()
    csv_pred = args.predictions
    dir_gt = args.ground_truth

    print("Loading prediction data...")
    df_pred = load_predictions(csv_pred)
    print(f"Loaded {len(df_pred)} predictions")
    print(f"Prediction columns: {list(df_pred.columns)}")
    
    print("\nLoading ground truth data...")
    df_gt = load_ground_truth(dir_gt)
    if df_gt is None:
        print("Could not load ground truth data")
        return
    
    print(f"Loaded {len(df_gt)} ground truth entries") 
    print(f"Ground truth columns: {list(df_gt.columns)}")
    
    print("\nComparing predictions with ground truth...")
    results, _ = compare_accuracy(df_pred, df_gt)
    
    if results:
        print_results(results)
    else:
        print("Could not compare data - check column names and data format")

if __name__ == "__main__":
    main()
