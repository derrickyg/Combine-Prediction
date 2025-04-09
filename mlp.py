import pandas as pd
import numpy as np
import os

def load_data():
    """Load the preprocessed data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'combined.csv')
    return pd.read_csv(data_path)

def show_top_rookie_metrics(n=30):
    df = load_data()
    print(df.describe())


    for metric in ['ROOKIE_SCORE']:
        print(f"\nTop {n} Players by {metric}:")
        print("=" * 80)
        top_players = df.nlargest(n, metric)
        print(top_players[['PLAYER_NAME', 'DRAFT_YEAR', metric, 'POINTS', 'REBOUNDS', 'ASSISTS', 'MINUTES']].to_string(index=False))


if __name__ == "__main__":
    show_top_rookie_metrics(30) 