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
        'TEMP_PLAYER_ID', 'HEIGHT_WO_SHOES_INCHES', 'HEIGHT_W_SHOES_INCHES',
        'SEASON'
    ]
    
    # Only remove columns that exist in the dataframe
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if columns_to_remove:
        print(f"Removing {len(columns_to_remove)} columns...")
        df = df.drop(columns=columns_to_remove)
    
    # Double check for HEIGHT_W_SHOES and its variants
    height_columns = [col for col in df.columns if 'HEIGHT_W_SHOES' in col]
    if height_columns:
        print(f"Found additional height columns to remove: {height_columns}")
        df = df.drop(columns=height_columns)
    
    return df

def standardize_name(name):
    """
    Standardize a name by removing special characters and standardizing common variations
    """
    if pd.isna(name):
        return name
    
    # Convert to string if not already
    name = str(name).strip()
    
    # Common name variations to standardize
    name_variations = {
        'PJ': 'P J',
        'AJ': 'A J',
        'TJ': 'T J',
        'CJ': 'C J',
        'JJ': 'J J',
        'DJ': 'D J',
        'RJ': 'R J',
        'JD': 'J D',
        'JR': 'J R'
    }
    
    # Replace periods and standardize spacing
    name = name.replace('.', ' ')  # Replace periods with spaces (P.J. -> P J)
    name = name.replace('-', ' ')  # Replace hyphens with spaces
    name = name.replace("'", '')   # Remove apostrophes
    
    # Replace multiple spaces with a single space
    name = ' '.join(name.split())
    
    # Check for name variations
    for variation, standard in name_variations.items():
        # Check for the variation at the start of the name or as a whole word
        if name.startswith(variation + ' ') or name == variation:
            name = name.replace(variation, standard, 1)
    
    return name.strip()

def standardize_player_names(df):
    """
    Standardize player names to handle variations like PJ vs P.J.
    """
    print("Standardizing player names...")
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Track name changes for debugging
    name_changes = []
    
    # Standardize FIRST_NAME and LAST_NAME if they exist
    if 'FIRST_NAME' in df.columns:
        for idx, row in df.iterrows():
            original_name = row['FIRST_NAME']
            standardized_name = standardize_name(original_name)
            if original_name != standardized_name:
                name_changes.append(f"{original_name} -> {standardized_name}")
            df.at[idx, 'FIRST_NAME_STD'] = standardized_name
    
    if 'LAST_NAME' in df.columns:
        df['LAST_NAME_STD'] = df['LAST_NAME'].apply(standardize_name)
    
    # Standardize PLAYER_NAME if it exists
    if 'PLAYER_NAME' in df.columns:
        df['PLAYER_NAME_STD'] = df['PLAYER_NAME'].apply(standardize_name)
    
    # Print name changes for debugging
    if name_changes:
        print("\nName standardization changes:")
        for change in name_changes:
            print(f"  {change}")
    
    return df

def find_duplicates(df):
    """
    Find duplicate players based on standardized names and draft year
    """
    print("Finding duplicate players...")
    
    # Create a copy of the dataframe
    df = df.copy()
    
    # Group by standardized names and draft year
    if 'FIRST_NAME_STD' in df.columns and 'LAST_NAME_STD' in df.columns:
        # Group by standardized first and last name
        duplicates = df.groupby(['FIRST_NAME_STD', 'LAST_NAME_STD', 'DRAFT_YEAR']).size().reset_index(name='count')
        duplicates = duplicates[duplicates['count'] > 1]
        
        if not duplicates.empty:
            print(f"Found {len(duplicates)} players with duplicate entries")
            
            # For each set of duplicates, keep the row with the most complete data
            rows_to_drop = []
            
            for _, row in duplicates.iterrows():
                first_name = row['FIRST_NAME_STD']
                last_name = row['LAST_NAME_STD']
                draft_year = row['DRAFT_YEAR']
                
                # Get all rows for this player
                player_rows = df[(df['FIRST_NAME_STD'] == first_name) & 
                                (df['LAST_NAME_STD'] == last_name) & 
                                (df['DRAFT_YEAR'] == draft_year)]
                
                # Count non-null values in each row
                non_null_counts = player_rows.count(axis=1)
                
                # Find the row with the most non-null values
                best_row_idx = non_null_counts.idxmax()
                
                # Add other rows to the drop list
                for idx in player_rows.index:
                    if idx != best_row_idx:
                        rows_to_drop.append(idx)
            
            # Drop the duplicate rows
            if rows_to_drop:
                print(f"Dropping {len(rows_to_drop)} duplicate rows")
                df = df.drop(rows_to_drop)
    
    # Special case: Check for PJ Washington specifically
    pj_washington_rows = df[df['LAST_NAME'].str.contains('Washington', na=False) & 
                           (df['FIRST_NAME'].str.contains('PJ', na=False) | 
                            df['FIRST_NAME'].str.contains('P.J', na=False))]
    
    if len(pj_washington_rows) > 1:
        print(f"Found {len(pj_washington_rows)} entries for PJ Washington")
        # Keep the row with the most non-null values
        non_null_counts = pj_washington_rows.count(axis=1)
        best_row_idx = non_null_counts.idxmax()
        
        # Drop other rows
        for idx in pj_washington_rows.index:
            if idx != best_row_idx:
                print(f"Dropping duplicate PJ Washington row with index {idx}")
                df = df.drop(idx)
    
    # Check for players with the same rookie stats across different seasons
    rookie_stats_columns = [
        'GAMES_PLAYED', 'MINUTES', 'POINTS', 'REBOUNDS', 'ASSISTS',
        'STEALS', 'BLOCKS', 'TURNOVERS', 'FIELD_GOAL_PCT', 'THREE_POINT_PCT',
        'FREE_THROW_PCT'
    ]
    
    # Filter to only include columns that exist in the dataframe
    existing_rookie_columns = [col for col in rookie_stats_columns if col in df.columns]
    
    if existing_rookie_columns:
        # Group by standardized names and rookie stats
        name_columns = ['FIRST_NAME_STD', 'LAST_NAME_STD'] if 'FIRST_NAME_STD' in df.columns else ['FIRST_NAME', 'LAST_NAME']
        
        # Create a key for grouping based on name and rookie stats
        df['rookie_stats_key'] = df[name_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        for col in existing_rookie_columns:
            df['rookie_stats_key'] += df[col].astype(str)
        
        # Find duplicates based on the rookie stats key
        duplicate_keys = df['rookie_stats_key'].value_counts()
        duplicate_keys = duplicate_keys[duplicate_keys > 1]
        
        if not duplicate_keys.empty:
            print(f"Found {len(duplicate_keys)} players with identical rookie stats across different seasons")
            
            # For each set of duplicates, keep the row with the earliest draft year
            rows_to_drop = []
            
            for key in duplicate_keys.index:
                # Get all rows with this key
                key_rows = df[df['rookie_stats_key'] == key]
                
                # Sort by draft year
                key_rows = key_rows.sort_values('DRAFT_YEAR')
                
                # Keep the first row (earliest draft year)
                keep_idx = key_rows.index[0]
                
                # Add other rows to the drop list
                for idx in key_rows.index[1:]:
                    player_name = f"{key_rows.loc[idx, 'FIRST_NAME']} {key_rows.loc[idx, 'LAST_NAME']}"
                    print(f"Dropping duplicate {player_name} row with index {idx} (same rookie stats as draft year {key_rows.loc[keep_idx, 'DRAFT_YEAR']})")
                    rows_to_drop.append(idx)
            
            # Drop the duplicate rows
            if rows_to_drop:
                print(f"Dropping {len(rows_to_drop)} duplicate rows with identical rookie stats")
                df = df.drop(rows_to_drop)
        
        # Drop the temporary key column
        df = df.drop(columns=['rookie_stats_key'])
    
    # Drop the standardized name columns
    std_columns = [col for col in df.columns if col.endswith('_STD')]
    if std_columns:
        df = df.drop(columns=std_columns)
    
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
    
    # Standardize player names
    df = standardize_player_names(df)
    
    # Find and remove duplicates based on standardized names
    df = find_duplicates(df)
    
    # Remove duplicate rows (as a fallback)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['FIRST_NAME', 'LAST_NAME', 'DRAFT_YEAR'])
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
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
    base_columns = {}
    for col in df.columns:
        if col.endswith('_x') or col.endswith('_y'):
            base_name = col[:-2]  # Remove _x or _y suffix
            if base_name not in base_columns:
                base_columns[base_name] = []
            base_columns[base_name].append(col)
    
    # For each set of duplicate columns, keep the one with fewer NaN values
    columns_to_drop = []
    columns_to_rename = {}
    
    for base_name, duplicate_cols in base_columns.items():
        if len(duplicate_cols) > 1:
            # Count NaN values in each duplicate column
            nan_counts = {}
            for col in duplicate_cols:
                nan_counts[col] = df[col].isna().sum()
            
            # Find the column with the fewest NaN values
            best_col = min(nan_counts, key=nan_counts.get)
            
            # Add other columns to drop list
            for col in duplicate_cols:
                if col != best_col:
                    columns_to_drop.append(col)
            
            # Rename the best column to the base name
            columns_to_rename[best_col] = base_name
    
    # Drop duplicate columns
    if columns_to_drop:
        print(f"Dropping {len(columns_to_drop)} duplicate columns")
        df = df.drop(columns=columns_to_drop)
    
    # Rename columns
    if columns_to_rename:
        print(f"Renaming {len(columns_to_rename)} columns")
        df = df.rename(columns=columns_to_rename)
    
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
        'PLAYER_ID', 'PLAYER_NAME', 'FIRST_NAME', 'LAST_NAME', 'POSITION', 'DRAFT_YEAR', 'SEASON'
    ]
    
    anthropometric_columns = [
        'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN',
        'STANDING_REACH', 'BODY_FAT_PCT', 'HAND_LENGTH', 'HAND_WIDTH', 'WINGSPAN_INCHES'
    ]
    
    strength_agility_columns = [
        'VERTICAL_LEAP_MAX_REACH', 'VERTICAL_LEAP_NO_STEP',
        'LANE_AGILITY_TIME', 'SHUTTLE_RUN', 'THREE_QUARTER_SPRINT', 'STANDING_VERTICAL_LEAP',
        'MAX_VERTICAL_LEAP', 'MODIFIED_LANE_AGILITY_TIME', 'STANDING_REACH_INCHES'
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
    # Exclude PRA_PERMIN and HEIGHT_W_SHOES
    remaining_columns = [col for col in df.columns if col not in ordered_columns 
                        and 'HEIGHT_W_SHOES' not in col 
                        and col != 'PRA_PERMIN']
    ordered_columns = ordered_columns + remaining_columns
    
    # Reorder the dataframe
    df = df[ordered_columns]
    
    print(f"Organized columns into {len(id_columns)} ID columns, {len(anthropometric_columns)} anthropometric columns, "
          f"{len(strength_agility_columns)} strength/agility columns, and {len(rookie_stats_columns)} rookie stats columns")
    
    return df

def check_for_duplicates(df):
    """
    Check for and display duplicate players in the dataset
    """
    print("\nChecking for duplicate players...")
    
    # Create standardized name columns
    df = standardize_player_names(df)
    
    # Group by standardized names and draft year
    if 'FIRST_NAME_STD' in df.columns and 'LAST_NAME_STD' in df.columns:
        # Group by standardized first and last name
        duplicates = df.groupby(['FIRST_NAME_STD', 'LAST_NAME_STD', 'DRAFT_YEAR']).size().reset_index(name='count')
        duplicates = duplicates[duplicates['count'] > 1]
        
        if not duplicates.empty:
            print(f"Found {len(duplicates)} players with duplicate entries:")
            
            # Display the duplicate players
            for _, row in duplicates.iterrows():
                first_name = row['FIRST_NAME_STD']
                last_name = row['LAST_NAME_STD']
                draft_year = row['DRAFT_YEAR']
                
                # Get all rows for this player
                player_rows = df[(df['FIRST_NAME_STD'] == first_name) & 
                                (df['LAST_NAME_STD'] == last_name) & 
                                (df['DRAFT_YEAR'] == draft_year)]
                
                print(f"\n{first_name} {last_name} (Draft Year: {draft_year}):")
                for idx, player_row in player_rows.iterrows():
                    print(f"  - {player_row['FIRST_NAME']} {player_row['LAST_NAME']} (ID: {player_row['PLAYER_ID']})")
        else:
            print("No duplicate players found.")
    
    # Drop the standardized name columns
    std_columns = [col for col in df.columns if col.endswith('_STD')]
    if std_columns:
        df = df.drop(columns=std_columns)
    
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
    
    # Check for duplicates
    check_for_duplicates(processed_df)

    processed_df['ROOKIE_SCORE'] = (
    processed_df['POINTS']
    + 0.4 * processed_df['FIELD_GOAL_PCT'] * processed_df['POINTS']
    + 0.7 * processed_df['REBOUNDS']
    + 0.7 * processed_df['ASSISTS']
    + processed_df['STEALS']
    + processed_df['BLOCKS']
    - processed_df['TURNOVERS'])
    
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