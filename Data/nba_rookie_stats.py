import pandas as pd
import time
from nba_combine_data import NBACombineData

def get_rookie_season(year: int) -> str:
    """Convert draft year to rookie season"""
    return f"{year}-{str(year + 1)[-2:]}"

def main():
    # Load the combine data
    combine_df = pd.read_csv('nba_combine_2014_2023.csv')
    
    # Initialize the NBA data fetcher
    nba_data = NBACombineData()
    
    # Create a new DataFrame for rookie stats
    rookie_stats = []
    
    # Iterate through each player
    for _, row in combine_df.iterrows():
        player_id = row['PLAYER_ID']
        draft_year = row['DRAFT_YEAR']
        rookie_season = get_rookie_season(draft_year)
        
        print(f"Fetching rookie stats for player {row['PLAYER_NAME']} ({draft_year})...")
        
        # Get rookie stats
        stats = nba_data.get_rookie_stats(player_id, rookie_season)
        
        if stats:
            # Add player identification info
            stats['PLAYER_NAME'] = row['PLAYER_NAME']
            stats['DRAFT_YEAR'] = draft_year
            rookie_stats.append(stats)
        
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Create DataFrame from rookie stats
    if rookie_stats:
        rookie_df = pd.DataFrame(rookie_stats)
        
        # Convert numeric columns
        numeric_columns = [
            'GAMES_PLAYED', 'MINUTES', 'POINTS', 'REBOUNDS', 'ASSISTS',
            'STEALS', 'BLOCKS', 'TURNOVERS', 'FIELD_GOAL_PCT',
            'THREE_POINT_PCT', 'FREE_THROW_PCT', 'USAGE_RATE', 'PLUS_MINUS'
        ]
        
        for col in numeric_columns:
            if col in rookie_df.columns:
                rookie_df[col] = pd.to_numeric(rookie_df[col], errors='coerce')
        
        # Merge with combine data
        final_df = pd.merge(combine_df, rookie_df, 
                          on=['PLAYER_ID', 'PLAYER_NAME', 'DRAFT_YEAR'], 
                          how='left')
        
        # Save the combined dataset
        final_df.to_csv('nba_combine_and_rookie_stats.csv', index=False)
        print("\nCombined dataset saved to 'nba_combine_and_rookie_stats.csv'")
        
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(final_df.info())
        
        print("\nNumber of players with rookie stats per draft year:")
        print(final_df.groupby('DRAFT_YEAR')['PLAYER_ID'].count())
        
        print("\nAverage rookie year statistics:")
        print(final_df[numeric_columns].describe())
    else:
        print("No rookie stats were found.")

if __name__ == "__main__":
    main() 