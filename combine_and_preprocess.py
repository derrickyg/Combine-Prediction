import pandas as pd
import numpy as np
import os

def combine_datasets():
    """
    Combine the NBA combine and rookie stats data with strength/agility data into a single dataset
    """
    # Check if files exist
    combine_rookie_file = 'nba_combine_and_rookie_stats_preprocessed.csv'
    strength_agility_file = 'nba_combine_strength_agility_2014_2023.csv'
    
    # Check for combine and rookie stats file
    if not os.path.exists(combine_rookie_file):
        print(f"Error: {combine_rookie_file} not found.")
        return None
    
    # Check for strength/agility file
    if not os.path.exists(strength_agility_file):
        print(f"Error: {strength_agility_file} not found.")
        return None
    
    # Load datasets
    print("Loading datasets...")
    combine_rookie_df = pd.read_csv(combine_rookie_file)
    strength_agility_df = pd.read_csv(strength_agility_file)
    
    # Extract first and last names from the PLAYER column if it exists
    if 'PLAYER' in combine_rookie_df.columns:
        print("Extracting first and last names from PLAYER column...")
        combine_rookie_df['FIRST_NAME'] = combine_rookie_df['PLAYER'].apply(lambda x: x.split()[0] if isinstance(x, str) and len(x.split()) > 0 else None)
        combine_rookie_df['LAST_NAME'] = combine_rookie_df['PLAYER'].apply(lambda x: ' '.join(x.split()[1:]) if isinstance(x, str) and len(x.split()) > 1 else None)
    
    if 'PLAYER' in strength_agility_df.columns:
        print("Extracting first and last names from PLAYER column...")
        strength_agility_df['FIRST_NAME'] = strength_agility_df['PLAYER'].apply(lambda x: x.split()[0] if isinstance(x, str) and len(x.split()) > 0 else None)
        strength_agility_df['LAST_NAME'] = strength_agility_df['PLAYER'].apply(lambda x: ' '.join(x.split()[1:]) if isinstance(x, str) and len(x.split()) > 1 else None)
    
    # Identify common columns for merging
    merge_columns = ['FIRST_NAME', 'LAST_NAME', 'DRAFT_YEAR']
    
    # Check if all merge columns exist in both dataframes
    missing_columns = [col for col in merge_columns if col not in combine_rookie_df.columns or col not in strength_agility_df.columns]
    if missing_columns:
        print(f"Warning: Some merge columns are missing: {missing_columns}")
        print("Attempting to merge on PLAYER_ID and DRAFT_YEAR instead.")
        merge_columns = ['PLAYER_ID', 'DRAFT_YEAR']
        
        # Check if PLAYER_ID exists in both dataframes
        if 'PLAYER_ID' not in combine_rookie_df.columns or 'PLAYER_ID' not in strength_agility_df.columns:
            print("Warning: PLAYER_ID column not found in one or both dataframes. Attempting to merge on PLAYER and DRAFT_YEAR.")
            merge_columns = ['PLAYER', 'DRAFT_YEAR']
    
    # Merge the dataframes
    print(f"Merging combine/rookie data with strength and agility data on {merge_columns}...")
    merged_df = pd.merge(combine_rookie_df, strength_agility_df, on=merge_columns, how='outer')
    
    # Print merge statistics
    print(f"Combine/rookie data: {len(combine_rookie_df)} rows")
    print(f"Strength/agility data: {len(strength_agility_df)} rows")
    print(f"Merged data: {len(merged_df)} rows")
    
    return merged_df

def remove_columns(df):
    """
    Remove specified columns from the dataset
    """
    columns_to_remove = [
        'HEIGHT_WO_SHOES_FT_IN', 'HEIGHT_W_SHOES_FT_IN', 
        'WINGSPAN_FT_IN', 'STANDING_REACH_FT_IN', 
        'BENCH_PRESS', 'SPOT_FIFTEEN_CORNER_LEFT', 
        'SPOT_FIFTEEN_BREAK_LEFT', 'SPOT_FIFTEEN_TOP_KEY', 
        'SPOT_FIFTEEN_BREAK_RIGHT', 'SPOT_FIFTEEN_CORNER_RIGHT', 
        'SPOT_COLLEGE_CORNER_LEFT', 'SPOT_COLLEGE_BREAK_LEFT', 
        'SPOT_COLLEGE_TOP_KEY', 'SPOT_COLLEGE_BREAK_RIGHT', 
        'SPOT_COLLEGE_CORNER_RIGHT', 'SPOT_NBA_CORNER_LEFT', 
        'SPOT_NBA_BREAK_LEFT', 'SPOT_NBA_TOP_KEY', 
        'SPOT_NBA_BREAK_RIGHT', 'SPOT_NBA_CORNER_RIGHT', 
        'OFF_DRIB_FIFTEEN_BREAK_LEFT', 'OFF_DRIB_FIFTEEN_TOP_KEY', 
        'OFF_DRIB_FIFTEEN_BREAK_RIGHT', 'OFF_DRIB_COLLEGE_BREAK_LEFT', 
        'OFF_DRIB_COLLEGE_TOP_KEY', 'OFF_DRIB_COLLEGE_BREAK_RIGHT', 
        'ON_MOVE_FIFTEEN', 'ON_MOVE_COLLEGE', 'HEIGHT_W_SHOES',
        'TEMP_PLAYER_ID', 'HEIGHT_WO_SHOES_INCHES', 'HEIGHT_W_SHOES_INCHES'
    ]
    
    # Only remove columns that exist in the dataframe
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if columns_to_remove:
        print(f"Removing {len(columns_to_remove)} columns...")
        df = df.drop(columns=columns_to_remove)
    else:
        print("No specified columns found to remove.")
    
    return df

def preprocess_data(df):
    """
    Preprocess the combined dataset
    """
    print("Preprocessing data...")
    
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Drop rows where all values are NaN
    df = df.dropna(axis=0, how='all')
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = initial_rows - len(df)
    print(f"Removed {removed_rows} duplicate rows")
    
    # Filter out players without rookie year stats
    rookie_stats_columns = [
        'GAMES_PLAYED', 'MINUTES', 'POINTS', 'REBOUNDS', 'ASSISTS',
        'STEALS', 'BLOCKS', 'TURNOVERS', 'FIELD_GOAL_PCT', 'THREE_POINT_PCT',
        'FREE_THROW_PCT'
    ]
    
    # Check which rookie stats columns exist in the dataframe
    existing_rookie_columns = [col for col in rookie_stats_columns if col in df.columns]
    
    if existing_rookie_columns:
        initial_rows = len(df)
        # Keep only rows where at least one rookie stat is not NaN
        df = df.dropna(subset=existing_rookie_columns, how='all')
        removed_rows = initial_rows - len(df)
        print(f"Removed {removed_rows} players without rookie year stats")
    else:
        print("Warning: No rookie stats columns found in the dataset")
    
    # Identify numeric columns dynamically
    numeric_columns = identify_numeric_columns(df)
    
    # Convert numeric columns to float
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where all numeric columns are NaN
    if numeric_columns:
        initial_rows = len(df)
        df = df.dropna(subset=numeric_columns, how='all')
        removed_rows = initial_rows - len(df)
        print(f"Removed {removed_rows} rows with all NaN numeric values")
    
    return df

def identify_numeric_columns(df):
    """
    Identify columns that should be treated as numeric based on their content
    """
    # List of potential numeric columns from all datasets
    potential_numeric_columns = [
        # Anthropometric measurements
        'HEIGHT_W_SHOES', 'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN',
        'STANDING_REACH', 'BODY_FAT_PCT', 'HAND_LENGTH', 'HAND_WIDTH',
        
        # Strength and agility measurements
        'BENCH_PRESS', 'VERTICAL_LEAP_MAX_REACH', 'VERTICAL_LEAP_NO_STEP',
        'LANE_AGILITY_TIME', 'SHUTTLE_RUN', 'THREE_QUARTER_SPRINT',
        
        # Rookie stats
        'GAMES_PLAYED', 'MINUTES', 'POINTS', 'REBOUNDS', 'ASSISTS',
        'STEALS', 'BLOCKS', 'TURNOVERS', 'FIELD_GOAL_PCT', 'THREE_POINT_PCT',
        'FREE_THROW_PCT', 'USAGE_RATE', 'PLUS_MINUS',
        
        # Additional columns that might be numeric
        'PLAYER_ID', 'DRAFT_YEAR'
    ]
    
    # Filter to only include columns that exist in the dataframe
    existing_columns = [col for col in potential_numeric_columns if col in df.columns]
    
    # Additional check: look for columns that contain numeric data
    numeric_columns = []
    for col in df.columns:
        # Skip columns we've already identified
        if col in existing_columns:
            numeric_columns.append(col)
            continue
            
        # Check if column contains numeric data
        try:
            # Try to convert a sample of non-null values to numeric
            sample = df[col].dropna().head(100)
            if not sample.empty:
                pd.to_numeric(sample, errors='raise')
                numeric_columns.append(col)
        except:
            # If conversion fails, column is not numeric
            pass
    
    print(f"Identified {len(numeric_columns)} numeric columns")
    return numeric_columns

def clean_duplicate_columns(df):
    """
    Clean up duplicate columns with _x and _y suffixes
    """
    print("Cleaning up duplicate columns...")
    
    # Find columns with _x and _y suffixes
    x_columns = [col for col in df.columns if col.endswith('_x')]
    y_columns = [col for col in df.columns if col.endswith('_y')]
    
    # Create a mapping of base column names to their _x and _y versions
    column_mapping = {}
    for x_col in x_columns:
        base_col = x_col[:-2]  # Remove _x suffix
        y_col = base_col + '_y'
        if y_col in y_columns:
            column_mapping[base_col] = (x_col, y_col)
    
    # For each pair of duplicate columns, keep the one with fewer NaN values
    columns_to_drop = []
    for base_col, (x_col, y_col) in column_mapping.items():
        # Count NaN values in each column
        x_nan_count = df[x_col].isna().sum()
        y_nan_count = df[y_col].isna().sum()
        
        # Keep the column with fewer NaN values
        if x_nan_count <= y_nan_count:
            # Rename x_col to base_col and drop y_col
            df = df.rename(columns={x_col: base_col})
            columns_to_drop.append(y_col)
        else:
            # Rename y_col to base_col and drop x_col
            df = df.rename(columns={y_col: base_col})
            columns_to_drop.append(x_col)
    
    # Drop the columns we decided to remove
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Cleaned up {len(columns_to_drop)} duplicate columns")
    else:
        print("No duplicate columns found")
    
    return df

def organize_columns(df):
    """
    Organize columns in a strategic order:
    1. Identification info
    2. Anthropometric measurements
    3. Strength and agility measurements
    4. Rookie stats
    """
    print("Organizing columns...")
    
    # Define column groups
    id_columns = [
        'PLAYER_ID', 'PLAYER_NAME', 'FIRST_NAME', 'LAST_NAME', 'POSITION', 'DRAFT_YEAR'
    ]
    
    anthropometric_columns = [
         'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN',
        'STANDING_REACH', 'BODY_FAT_PCT', 'HAND_LENGTH', 'HAND_WIDTH'
    ]
    
    strength_agility_columns = [
        'BENCH_PRESS', 'VERTICAL_LEAP_MAX_REACH', 'VERTICAL_LEAP_NO_STEP',
        'LANE_AGILITY_TIME', 'SHUTTLE_RUN', 'THREE_QUARTER_SPRINT', 'STANDING_VERTICAL_LEAP',
        'MAX_VERTICAL_LEAP', 'MODIFIED_LANE_AGILITY_TIME'
    ]
    
    rookie_stats_columns = [
        'GAMES_PLAYED', 'MINUTES', 'POINTS', 'REBOUNDS', 'ASSISTS',
        'STEALS', 'BLOCKS', 'TURNOVERS', 'FIELD_GOAL_PCT', 'THREE_POINT_PCT',
        'FREE_THROW_PCT', 'USAGE_RATE', 'PLUS_MINUS'
    ]
    
    # Filter to only include columns that exist in the dataframe
    id_columns = [col for col in id_columns if col in df.columns]
    anthropometric_columns = [col for col in anthropometric_columns if col in df.columns]
    strength_agility_columns = [col for col in strength_agility_columns if col in df.columns]
    rookie_stats_columns = [col for col in rookie_stats_columns if col in df.columns]
    
    # Combine all column groups
    ordered_columns = id_columns + anthropometric_columns + strength_agility_columns + rookie_stats_columns
    
    # Add any remaining columns that weren't in our predefined groups
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    ordered_columns = ordered_columns + remaining_columns
    
    # Reorder the dataframe
    df = df[ordered_columns]
    
    print(f"Organized columns into {len(id_columns)} ID columns, {len(anthropometric_columns)} anthropometric columns, "
          f"{len(strength_agility_columns)} strength/agility columns, and {len(rookie_stats_columns)} rookie stats columns")
    
    return df

def main():
    # Combine datasets
    combined_df = combine_datasets()
    
    if combined_df is None or combined_df.empty:
        print("Failed to combine datasets.")
        return
    
    # Remove specified columns
    combined_df = remove_columns(combined_df)
    
    # Clean up duplicate columns
    combined_df = clean_duplicate_columns(combined_df)
    
    # Preprocess the data
    processed_df = preprocess_data(combined_df)
    
    # Organize columns
    processed_df = organize_columns(processed_df)
    
    # Save the processed data
    output_file = 'combined.csv'
    processed_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to '{output_file}'")
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(processed_df.info())
    
    # Display number of players per year
    print("\nNumber of players per draft year:")
    print(processed_df['DRAFT_YEAR'].value_counts().sort_index())
    
    # Display the first few rows
    print("\nFirst few rows of the dataset:")
    print(processed_df.head())

if __name__ == "__main__":
    main() 