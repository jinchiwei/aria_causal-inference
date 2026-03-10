import pandas as pd
import os

# Test the encoding logic for Luke files
def test_luke_encoding():
    print("Testing Luke file encoding logic...")

    # Simulate microhemorrhage data as it might appear in Luke files
    test_data = {
        'Microhemorrhage': [
            '0-4',
            '0-4 ?',
            '5-10',
            '5-10 maybe',
            '>10',
            '>10 confirmed',
            'Missing',
            'missing',
            'None',
            'unclear',
            '1',
            ''
        ]
    }

    df = pd.DataFrame(test_data)
    print("\nOriginal data:")
    print(df)

    # Apply the encoding logic
    processed_values = []
    for val in df['Microhemorrhage']:
        val_str = str(val).strip()
        # Check for histogram values
        if "0-4" in val_str:
            processed_values.append(0)
        elif "5-10" in val_str:
            processed_values.append(1)
        elif ">10" in val_str:
            processed_values.append(2)
        # Handle Missing values
        elif "missing" in val_str.lower():
            processed_values.append(-1)
        # Handle None/nan values
        elif val_str.lower() in ["none", "nan", "null", ""]:
            processed_values.append(0)
        # Handle unclear values
        elif val_str.lower() in ["unclear", "?"] or "not co" in val_str.lower():
            processed_values.append(-1)
        else:
            # Try to keep numeric values as is, otherwise 0
            try:
                processed_values.append(int(float(val_str)))
            except:
                processed_values.append(0)

    df['Encoded'] = processed_values
    print("\nEncoded data:")
    print(df)

    # Test Luke 2 "Yes" encoding
    print("\n" + "="*50)
    print("Testing Luke 2 'Yes' encoding...")

    test_data_2 = {
        'ARIA-E': ['Yes', 'yes', 'YES', 'None', '1', '0', 'unclear', 'Missing'],
        'ARIA-H': ['Yes', 'No', '1', '0', 'unclear', '', 'none', 'missing']
    }

    df2 = pd.DataFrame(test_data_2)
    print("\nOriginal Luke 2 data:")
    print(df2)

    for col in ['ARIA-E', 'ARIA-H']:
        # Convert to string first
        df2[col] = df2[col].astype(str)

        # Create a copy for processing
        processed_values = []
        for val in df2[col]:
            val_str = str(val).strip()
            # Check for "Yes" values
            if val_str.lower() in ["yes"]:
                processed_values.append(1)
            # Handle Missing values
            elif "missing" in val_str.lower():
                processed_values.append(-1)
            # Handle None/nan/No values
            elif val_str.lower() in ["none", "nan", "null", "", "no"]:
                processed_values.append(0)
            # Handle unclear values
            elif val_str.lower() in ["unclear", "?"] or "not co" in val_str.lower():
                processed_values.append(-1)
            else:
                # Try to keep numeric values as is, otherwise 0
                try:
                    processed_values.append(int(float(val_str)))
                except:
                    processed_values.append(0)

        df2[col] = processed_values

    print("\nEncoded Luke 2 data:")
    print(df2)

if __name__ == "__main__":
    test_luke_encoding()