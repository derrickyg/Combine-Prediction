import pandas as pd
import numpy as np
import os

def load_data():
    """Load the preprocessed data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'nba_combine_and_rookie_stats_preprocessed.csv')
    return pd.read_csv(data_path)

def show_top_pra_players(n=30):
    """Display the top n players by PRA_PERMIN"""
    # Load the data
    df = load_data()
    
    # Sort by PRA_PERMIN in descending order
    top_players = df.nlargest(n, 'PRA_PERMIN')
    
    # Select relevant columns for display
    display_columns = [
        'PLAYER_NAME', 'DRAFT_YEAR', 'PRA_PERMIN',
        'POINTS', 'REBOUNDS', 'ASSISTS', 'MINUTES'
    ]
    
    # Display the results
    print(f"\nTop {n} Players by PRA_PERMIN:")
    print("=" * 80)
    print(top_players[display_columns].to_string(index=False))
    print("=" * 80)
    
    # Display some statistics about these players
    print("\nStatistics for Top Players:")
    print("-" * 40)
    print(f"Average PRA_PERMIN: {top_players['PRA_PERMIN'].mean():.3f}")
    print(f"Average Minutes: {top_players['MINUTES'].mean():.2f}")
    print(f"Average Points: {top_players['POINTS'].mean():.2f}")
    print(f"Average Rebounds: {top_players['REBOUNDS'].mean():.2f}")
    print(f"Average Assists: {top_players['ASSISTS'].mean():.2f}")
    
    # Show distribution by draft year
    print("\nDistribution by Draft Year:")
    print("-" * 40)
    print(top_players['DRAFT_YEAR'].value_counts().sort_index())

if __name__ == "__main__":
    show_top_pra_players(30) 