import os
from pathlib import Path

import pandas as pd

def combine_xlsx_files():
    # Define diagnostic columns that should be handled specially
    diagnostic_columns = ["ARIA-E", "ARIA-H", "Edema", "Effusion", "Microhemorrhage", "Superficial Siderosis"]

    labeled_dir = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled")
    preferred_order = [
        "ARIA Labeling - Ali.xlsx",
        "ARIA Labeling - Luke.xlsx",
        "ARIA Labeling - Michael.xlsx",
        "ARIA Labeling - Michael 2_annotated.xlsx",
        "ARIA Labeling - Luke 2_annotated.xlsx",
    ]
    existing = {p.name: p for p in labeled_dir.glob("ARIA Labeling - *.xlsx")}
    files = [str(existing[n]) for n in preferred_order if n in existing]
    # Include any other annotator files (stable ordering) but exclude generated outputs.
    extras = sorted(
        str(p)
        for n, p in existing.items()
        if n not in preferred_order and n not in {"combined_annotations.xlsx"} and not n.startswith("differences-")
    )
    files.extend(extras)
    if not files:
        print(f"No annotator XLSX files found in: {labeled_dir}")
        return None

    print("Analyzing column structure...")
    print("=" * 50)

    # Step 1: Read all files and analyze columns
    dataframes = {}
    all_columns = {}
    
    for file_path in files:
        if os.path.exists(file_path):
            try:
                # First read to get column names
                temp_df = pd.read_excel(file_path, nrows=0)

                # Create dtype dict - diagnostic columns as string, others as default
                dtype_dict = {}
                for col in temp_df.columns:
                    if col in diagnostic_columns:
                        dtype_dict[col] = str

                # Read with specified dtypes
                df = pd.read_excel(file_path, dtype=dtype_dict)

                # Special handling for Luke files
                is_luke_file = "Luke" in os.path.basename(file_path)
                is_luke_2_file = "Luke 2" in os.path.basename(file_path)

                # Replace "None"/"none" and NaN with 0, "unclear" with -1 for diagnostic columns
                for col in diagnostic_columns:
                    if col in df.columns:
                        # Convert to string first to handle mixed types
                        df[col] = df[col].astype(str)

                        # Special handling for Luke files - Microhemorrhage histogram encoding
                        if is_luke_file and col == "Microhemorrhage":
                            # Create a copy for processing
                            processed_values = []
                            for val in df[col]:
                                val_str = str(val).strip()
                                # Check for histogram values
                                if "0-4" in val_str:
                                    processed_values.append(0)
                                elif "5-10" in val_str or "5-9" in val_str:
                                    processed_values.append(1)
                                elif ">10" in val_str:
                                    processed_values.append(2)
                                # Handle Missing values
                                elif "missing" in val_str.lower():
                                    processed_values.append(-1)
                                # Handle None/nan values
                                elif val_str.lower() in ["none", "nan", "null", ""]:
                                    processed_values.append(0)
                                # Handle unclear/possible values
                                elif val_str.lower() in ["unclear", "?", "possible"] or "not co" in val_str.lower():
                                    processed_values.append(-1)
                                # Handle "Mild" as present (1)
                                elif "mild" in val_str.lower():
                                    processed_values.append(1)
                                # Handle negated findings (e.g., "no hemorrhage")
                                elif val_str.lower().startswith("no "):
                                    processed_values.append(0)
                                else:
                                    # Try to keep numeric values as is
                                    try:
                                        processed_values.append(int(float(val_str)))
                                    except:
                                        # Descriptive text that isn't a known absent/unclear pattern = present
                                        if len(val_str) > 3:
                                            processed_values.append(1)
                                        else:
                                            processed_values.append(0)

                            df[col] = processed_values

                        # Special handling for Luke files (non-Microhemorrhage columns) - apply histogram encoding to all diagnostic columns
                        elif is_luke_file:
                            # Create a copy for processing
                            processed_values = []
                            for val in df[col]:
                                val_str = str(val).strip()
                                # Check for histogram values (in case other columns also have them)
                                if "0-4" in val_str:
                                    processed_values.append(0)
                                elif "5-10" in val_str or "5-9" in val_str:
                                    processed_values.append(1)
                                elif ">10" in val_str:
                                    processed_values.append(2)
                                # Handle "Yes" variants (Yes, Yes (1), Yes (2), etc.) as present
                                elif val_str.lower().startswith("yes"):
                                    processed_values.append(1)
                                # Handle "Mild" as present (1)
                                elif "mild" in val_str.lower():
                                    processed_values.append(1)
                                # Handle "focal area" patterns as present (1)
                                elif "focal area" in val_str.lower():
                                    processed_values.append(1)
                                # Handle Missing values
                                elif "missing" in val_str.lower():
                                    processed_values.append(-1)
                                # Handle None/nan values
                                elif val_str.lower() in ["none", "nan", "null", ""]:
                                    processed_values.append(0)
                                # Handle unclear/possible values
                                elif val_str.lower() in ["unclear", "unclear.", "?", "possible"] or "not co" in val_str.lower():
                                    processed_values.append(-1)
                                # Handle negated findings (e.g., "no effusion")
                                elif val_str.lower().startswith("no "):
                                    processed_values.append(0)
                                else:
                                    # Try to keep numeric values as is
                                    try:
                                        processed_values.append(int(float(val_str)))
                                    except:
                                        # Descriptive text that isn't a known absent/unclear pattern = present
                                        if len(val_str) > 3:
                                            processed_values.append(1)
                                        else:
                                            processed_values.append(0)

                            df[col] = processed_values

                        # Standard handling for other files (non-Luke files)
                        else:
                            # Replace "Missing" with -1 (case-insensitive)
                            df[col] = df[col].replace(to_replace=r'(?i).*missing.*', value=-1, regex=True)
                            # Replace string versions of None with 0
                            df[col] = df[col].replace(["None", "none", "NONE", "NOne", "nan"], 0)
                            # Replace case-insensitive "unclear" with -1
                            df[col] = df[col].replace(["unclear", "Unclear", "UNCLEAR", "Unclear.", "?"], -1)
                            # Replace "Not co*" pattern with -1 (case-insensitive)
                            df[col] = df[col].replace(to_replace=r'(?i)^not co.*$', value=-1, regex=True)
                            # Replace "BASELINE" or "Can't apply" patterns with -1 (case-insensitive)
                            df[col] = df[col].replace(to_replace=r"(?i).*(baseline|can't apply).*", value=-1, regex=True)
                            # Convert to numeric where possible
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                dataframes[file_path] = df
                columns = list(df.columns)
                
                print(f"\nFile: {os.path.basename(file_path)}")
                print(f"Rows: {len(df)}, Columns: {len(columns)}")
                
                # Track column positions across files
                for i, col in enumerate(columns):
                    if col not in all_columns:
                        all_columns[col] = {}
                    all_columns[col][file_path] = i
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return None
        else:
            print(f"File not found: {file_path}")
            return None

    # Step 2: Identify common columns (appear in ALL files) vs unique columns
    first_file_path = files[0]
    first_file_columns = list(dataframes[first_file_path].columns)

    common_columns = []
    unique_columns = []

    # Find columns that appear in ALL files, preserve order from first file
    for col in first_file_columns:
        if len(all_columns[col]) == len(files):  # Column appears in all files
            common_columns.append(col)

    # Find remaining columns that don't appear in all files
    for col, file_positions in all_columns.items():
        if len(file_positions) < len(files):  # Column doesn't appear in all files
            unique_columns.append(col)
    
    print(f"\nCommon columns (appear in ALL files): {len(common_columns)}")
    for i, col in enumerate(common_columns):
        print(f"  {i}: {col}")

    print(f"\nUnique columns (not in all files): {len(unique_columns)}")
    for col in sorted(unique_columns):
        print(f"  {col}")
    
    # Step 3: Combine dataframes
    print(f"\nCombining files...")
    combined_dfs = []
    
    for file_path, df in dataframes.items():
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        # Start with common columns that exist in this file
        combined_df = pd.DataFrame()
        for col in common_columns:
            if col in df.columns:
                combined_df[col] = df[col]

        # Add unique columns from this file
        for col in df.columns:
            if col in unique_columns:
                combined_df[col] = df[col]
        
        # Add a source column to track which file each row came from
        combined_df['source_file'] = filename
        
        combined_dfs.append(combined_df)
    
    # Step 4: Concatenate all dataframes
    final_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
    
    # Reorder columns: common columns first, then unique columns, then source
    final_columns = common_columns.copy()
    for col in sorted(unique_columns):
        if col in final_df.columns:
            final_columns.append(col)
    final_columns.append('source_file')
    
    final_df = final_df[final_columns]
    
    print(f"\nCombined dataset:")
    print(f"Total rows: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")
    
    # Step 5: Save combined file
    output_path = "/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled/combined_annotations.xlsx"
    final_df.to_excel(output_path, index=False)
    print(f"\nCombined file saved to: {output_path}")
    
    return final_df

if __name__ == "__main__":
    result = combine_xlsx_files()
    if result is not None:
        print(f"\nFirst few rows of combined data:")
        print(result.head())
