# type: ignore
from pathlib import Path

import pandas as pd


def main():
    dir_data = Path(r'/data/rauschecker1/jkw/projects/alzheimers/aria/aria_prediction_modeling/data')
    mpower_path = dir_data / 'search_aria.xlsx'
    mpower_df = pd.read_excel(mpower_path)
    
    # Create the format expected by download.sh: accession_number, rename_subjectID, rename_sessionID
    # Use Patient MRN as subjectID and Accession Number as sessionID
    mpower_df['session_id'] = mpower_df['Accession Number']  # Use accession as session ID
    mpower_df = mpower_df[['Accession Number', 'Patient MRN', 'session_id']]
    
    # Save without header since download.sh doesn't expect one
    mpower_df.to_csv(dir_data / 'ucsf_aria' / '2025-07-08_aria-accessions.csv', index=False, header=False)


if __name__ == '__main__':
    main()


# def main():
#     dir_data = Path(r'/data/rauschecker1/jkw/projects/alzheimers/aria/aria_prediction_modeling/data')
#     mpower_path = dir_data / 'search_aria.xlsx'
#     mpower_df = pd.read_excel(mpower_path)
    
#     # Sort by Patient MRN and Exam Completed Date to get chronological order within each patient
#     mpower_df = mpower_df.sort_values(['Patient MRN', 'Exam Completed Date'])
    
#     # Create sequential session IDs for each patient (001, 002, 003, etc.)
#     mpower_df['session_id'] = mpower_df.groupby('Patient MRN').cumcount() + 1
#     mpower_df['session_id'] = mpower_df['session_id'].apply(lambda x: f'{x:03d}')
    
#     # Create the format expected by download.sh: accession_number, rename_subjectID, rename_sessionID
#     # Use Patient MRN as subjectID and sequential session_id based on chronological order
#     mpower_df = mpower_df[['Accession Number', 'Patient MRN', 'session_id']]
    
#     # Save without header since download.sh doesn't expect one
#     mpower_df.to_csv(dir_data / 'ucsf_aria' / '2025-07-08_aria-accessions.csv', index=False, header=False)
