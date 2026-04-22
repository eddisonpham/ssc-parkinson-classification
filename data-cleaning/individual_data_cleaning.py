import pandas as pd

def convert_numerical_strings(df):
    """Convert numerical string columns to integer or float."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Try to convert to numeric
                converted = pd.to_numeric(df[col])
                # Check if all values are integers
                if converted.dropna().apply(lambda x: x == int(x)).all():
                    df[col] = converted.astype('Int64')
                else:
                    df[col] = converted
                print(f"Converted '{col}' to numeric")
            except (ValueError, TypeError):
                pass
    return df

def convert_date_strings(df):
    """Convert date string columns (M/D/YYYY format) to datetime objects."""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                converted = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
                # Only convert if all non-NA values were successfully parsed
                non_na_original = df[col].notna().sum()
                non_na_converted = converted.notna().sum()
                if non_na_original > 0 and non_na_original == non_na_converted:
                    df[col] = converted
                    print(f"Converted '{col}' to datetime")
            except (ValueError, TypeError):
                pass
    return df

def checking(df):
    """Checking the column names and type"""
    print()
    print("===============================================================================")
    print()
    print("Column names and data type:")
    for col in df.columns:
        print("==  " + col)
        print(df[col].dtype)
        print(df[col].dropna().unique())
        print()
    print("===============================================================================")
    print()

def split_and_create_boolean_columns(df, col):
    """Split comma-separated values in a column and create boolean columns."""

    if col not in df.columns:
        print(col + " is not in df")
        return df
         
    # Collect all unique values from the column
    all_values = set()
    for value in df[col].dropna():
        all_values.update([v.strip() for v in str(value).split(',')])

    # Create boolean columns for each unique value
    for val in sorted(all_values):
        new_col = val + "_" + col
        df[new_col] = df[col].fillna('').apply(
            lambda x: 1 if val in str(x).split(',') else 0
        )
    return df

if __name__ == "__main__":
    filepaths = [
        "epidemiological.csv",
        "fatigue_severity_scale.csv",
        "mds-updrs.csv",
        "mds-updrs-1.csv",
        "schwab_&_england.csv"
        ]
    delete_columns = {
        "epidemiological.csv" : [
            "Event Name", 
            "How is the questionnaire completed?     Comment le questionnaire est-il rempli?"
            ],
        "fatigue_severity_scale.csv" : [
            "How is the questionnaire completed?     Comment le questionnaire est-il rempli?",
            "Divided by 9"
            ],
        "mds-updrs.csv" : [
            "Assessment completed by:     Évaluation complétée par:",
            "How was the MDS-UPDRS administered?   Comment le MDS-UPDRS a-t-il été administré?",
            "Is it UPDRS v1.2 (part3) legacy ?  S'agit il d'un héritage de UPDRS v1.2 (partie 3) ?",
            "Primary source of information:",
            "Primary source of information:",
            "Updrs_1_1 value",
            "Updrs_1_2 value",
            "Updrs_1_3",
            "Updrs_1_4 value",
            "Updrs_1_5 value",
            "Updrs_1_6 value",
            "Who is filling out this questionnaire (check the best answer):",
            "3.C1 Time of assessment:   Temps de l'évaluation:",
            "3.C1 Time last PD medications taken:   Heure de la dernière prise de médicaments pour PD:",
            "Updrs_3_1 value",
            "Updrs_3_2 value",
            "Updrs_3_3_neck value",
            "Updrs_3_3_rue value",
            "Updrs_3_3_lue value",
            "Updrs_3_3_rle value",
            "Updrs_3_3_lle value",
            "Updrs_3_4_r value",
            "Updrs_3_4_l value",
            "Updrs_3_5_r value",
            "Updrs_3_5_l value",
            "Updrs_3_6_r value",
            "Updrs_3_6_l value",
            "Updrs_3_7_r value",
            "Updrs_3_7_l value",
            "Updrs_3_8_r value",
            "Updrs_3_8_l value",
            "Updrs_3_9 value",
            "Updrs_3_10 value",
            "Updrs_3_11 value",
            "Updrs_3_12 value",
            "Updrs_3_13 value",
            "Updrs_3_14",
            "Updrs_3_15_r value",
            "Updrs_3_15_l value",
            "Updrs_3_16_r value",
            "Updrs_3_16_l value",
            "Updrs_3_17_rue value",
            "Updrs_3_17_lue value",
            "Updrs_3_17_rle value",
            "Updrs_3_17_lle value",
            "Updrs_3_17_lipjaw value",
            "Updrs_3_18 value",
            "Updrs_4_1 value",
            "Updrs_4_2 value",
            "Updrs_4_3 value",
            "Updrs_4_4 value",
            "Updrs_4_5 value",
            "Updrs_4_6 value",
        ],
        "mds-updrs-1.csv" : [
            "Assessment completed by:     Évaluation complétée par:",
            "How was the MDS-UPDRS administered?   Comment le MDS-UPDRS a-t-il été administré?",
            "Is it UPDRS v1.2 (part3) legacy ?  S'agit il d'un héritage de UPDRS v1.2 (partie 3) ?",
            "Primary source of information:",
            "Updrs_1_1 value",
            "Updrs_1_2 value",
            "Updrs_1_3",
            "Updrs_1_4 value",
            "Updrs_1_5 value",
            "Updrs_1_6 value",
            "Who is filling out this questionnaire (check the best answer):",
            "3.C1 Time of assessment:   Temps de l'évaluation:",
            "3.C1 Time last PD medications taken:   Heure de la dernière prise de médicaments pour PD:",
            "Updrs_3_1 value",
            "Updrs_3_2 value",
            "Updrs_3_3_neck value",
            "Updrs_3_3_rue value",
            "Updrs_3_3_lue value",
            "Updrs_3_3_rle value",
            "Updrs_3_3_lle value",
            "Updrs_3_4_r value",
            "Updrs_3_4_l value",
            "Updrs_3_5_r value",
            "Updrs_3_5_l value",
            "Updrs_3_6_r value",
            "Updrs_3_6_l value",
            "Updrs_3_7_r value",
            "Updrs_3_7_l value",
            "Updrs_3_8_r value",
            "Updrs_3_8_l value",
            "Updrs_3_9 value",
            "Updrs_3_10 value",
            "Updrs_3_11 value",
            "Updrs_3_12 value",
            "Updrs_3_13 value",
            "Updrs_3_14",
            "Updrs_3_15_r value",
            "Updrs_3_15_l value",
            "Updrs_3_16_r value",
            "Updrs_3_16_l value",
            "Updrs_3_17_rue value",
            "Updrs_3_17_lue value",
            "Updrs_3_17_rle value",
            "Updrs_3_17_lle value",
            "Updrs_3_17_lipjaw value",
            "Updrs_3_18 value",
            "Updrs_4_1 value",
            "Updrs_4_2 value",
            "Updrs_4_3 value",
            "Updrs_4_4 value",
            "Updrs_4_5 value",
            "Updrs_4_6 value"
        ],
        "schwab_&_england.csv" : [
            "How is the questionnaire completed?     Comment le questionnaire est-il rempli?"
        ]
        }
    check_all_type_columns = {
        "epidemiological.csv" : [
            "1. Have you ever been diagnosed with:    1. Avez-vous déjà été diagnostiqué pour:", 
            "2b. Reason for admission:    2b. Raison de l'admission:", 
            "21. What type of exercise do you do?    21. Quel type d'exercice faites-vous?", 
            "22. Are you currently receiving any of the following therapies?    22. Suivez-vous présentement une ou plusieurs des thérapies suivantes?",
            "24. Do you currently use any of the following?    24. Consommez-vous les substances suivantes?", 
            "25a. What is the ethnicity of your biological father?    25a. Quelle est l'ethnicité de votre père biologique?", 
            "26a. What is the ethnicity of your biological mother?    26a. Quelle est l'ethnicité de votre mère biologique?"
            ],
        "fatigue_severity_scale.csv" : [],
        "mds-updrs-1.csv": [],
        "mds-updrs.csv": [],
        "schwab_&_england.csv" : []
    }

    for file in filepaths:
        df = pd.read_csv(file)
        checking(df)

        # strips newlines
        df.replace(r'\n|\r', ' ', regex=True)

        # change check-all responses to individual boolean features
        for column in check_all_type_columns[file]:
            df = split_and_create_boolean_columns(df, column)

        # drops entirely empty columns
        df.dropna(axis=1, how='all') 

        # feature selection
        df = df.drop(columns=delete_columns[file]) 

        convert_date_strings(df) # is this useful/necessary?
        convert_numerical_strings(df)
        
        # change all column names to filename_column name to ensure uniqueness
        df.columns = [f"{file.replace('.csv', '')}_{col}" for col in df.columns]

        checking(df)
        cleaned_filename = "cleaned_" + file
        df.to_csv(cleaned_filename, index=False)
