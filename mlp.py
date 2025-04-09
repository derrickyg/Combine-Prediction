import pandas as pd
import numpy as np
import os

def load_data():
    """Load the preprocessed data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'nba_combine_and_rookie_stats_preprocessed.csv')
    return pd.read_csv(data_path)

def show_top_rookie_metrics(n=30):
    df = load_data()

    df['ROOKIE_SCORE'] = (
    df['POINTS']
    + 0.4 * df['FIELD_GOAL_PCT'] * df['POINTS']
    + 0.7 * df['REBOUNDS']
    + 0.7 * df['ASSISTS']
    + df['STEALS']
    + df['BLOCKS']
    - df['TURNOVERS'])


    for metric in ['ROOKIE_SCORE']:
        print(f"\nTop {n} Players by {metric}:")
        print("=" * 80)
        top_players = df.nlargest(n, metric)
        print(top_players[['PLAYER_NAME', 'DRAFT_YEAR', metric, 'POINTS', 'REBOUNDS', 'ASSISTS', 'MINUTES']].to_string(index=False))


if __name__ == "__main__":
    show_top_rookie_metrics(30) 