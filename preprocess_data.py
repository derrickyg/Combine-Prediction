import pandas as pd
import numpy as np

def convert_height_to_inches(height_str):
    """
    Convert height string (e.g., "6' 11.5''") to total inches
    """
    if pd.isna(height_str):
        return np.nan
    
    try:
        # Split into feet and inches
        parts = height_str.split("'")
        feet = float(parts[0])
        inches = float(parts[1].replace('"', ''))
        
        # Convert to total inches
        total_inches = (feet * 12) + inches
        return total_inches
    except:
        return np.nan

def preprocess_data():
    # Read the data
    print("Loading data...")
    df = pd.read_csv('nba_combine_and_rookie_stats.csv')
    
    # Remove USAGE_RATE and PLUS_MINUS columns
    print("Removing USAGE_RATE and PLUS_MINUS columns...")
    df = df.drop(['USAGE_RATE', 'PLUS_MINUS'], axis=1, errors='ignore')
    
    # Remove rows where minutes are blank
    print("Removing rows with blank minutes...")
    df = df.dropna(subset=['MINUTES', 'POINTS'])
    
    # Convert height and reach measurements to inches
    print("Converting height and reach measurements to inches...")
    height_columns = [
        'HEIGHT_WO_SHOES_FT_IN',
        'HEIGHT_W_SHOES_FT_IN',
        'WINGSPAN_FT_IN',
        'STANDING_REACH_FT_IN'
    ]
    
    for col in height_columns:
        if col in df.columns:
            new_col_name = col.replace('_FT_IN', '_INCHES')
            df[new_col_name] = df[col].apply(convert_height_to_inches)
            print(f"Converted {col} to {new_col_name}")
    
    # Save the preprocessed data
    print("Saving preprocessed data...")
    df.to_csv('nba_combine_and_rookie_stats_preprocessed.csv', index=False)
    
    # Display information about the preprocessed dataset
    print("\nPreprocessed Dataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nNumber of rows in final dataset:", len(df))

if __name__ == "__main__":
    preprocess_data() 