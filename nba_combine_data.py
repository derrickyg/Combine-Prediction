import pandas as pd
import requests
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional
import time

class NBACombineData:
    def __init__(self):
        self.base_url = "https://stats.nba.com/stats"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/'
        }

    def get_combine_data(self, year: int) -> Optional[pd.DataFrame]:
        """
        Fetch NBA combine data for a specific year
        """
        try:
            endpoint = f"{self.base_url}/draftcombineplayeranthro"
            params = {
                'LeagueID': '00',
                'SeasonYear': str(year),
                'Height': 'No',
                'Weight': 'No',
                'Wingspan': 'No',
                'StandingReach': 'No',
                'BodyFatPct': 'No',
                'HandLength': 'No',
                'HandWidth': 'No'
            }
            
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract headers and rows
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            df['DRAFT_YEAR'] = year  # Add draft year column
            return df
            
        except Exception as e:
            print(f"Error fetching combine data for year {year}: {str(e)}")
            return None

    def get_rookie_stats(self, player_id: str, season: str) -> Optional[Dict]:
        """
        Fetch rookie year statistics for a player
        """
        try:
            # Get player profile stats
            endpoint = f"{self.base_url}/playerprofilev2"
            params = {
                'PlayerID': player_id,
                'Season': season,
                'SeasonType': 'Regular Season',
                'LeagueID': '00',
                'PerMode': 'PerGame'
            }
            
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data['resultSets'][0]['rowSet']:
                return None
            
            # Get the first row (season averages)
            stats = data['resultSets'][0]['rowSet'][0]
            headers = data['resultSets'][0]['headers']
            
            # Create a dictionary with relevant stats
            relevant_stats = {
                'PLAYER_ID': player_id,
                'SEASON': season,
                'GAMES_PLAYED': stats[headers.index('GP')],
                'MINUTES': stats[headers.index('MIN')],
                'POINTS': stats[headers.index('PTS')],
                'REBOUNDS': stats[headers.index('REB')],
                'ASSISTS': stats[headers.index('AST')],
                'STEALS': stats[headers.index('STL')],
                'BLOCKS': stats[headers.index('BLK')],
                'TURNOVERS': stats[headers.index('TOV')],
                'FIELD_GOAL_PCT': stats[headers.index('FG_PCT')],
                'THREE_POINT_PCT': stats[headers.index('FG3_PCT')],
                'FREE_THROW_PCT': stats[headers.index('FT_PCT')],
                'USAGE_RATE': stats[headers.index('USG_PCT')] if 'USG_PCT' in headers else None,
                'PLUS_MINUS': stats[headers.index('PLUS_MINUS')] if 'PLUS_MINUS' in headers else None
            }
            
            return relevant_stats
            
        except Exception as e:
            print(f"Error fetching rookie stats for player {player_id}: {str(e)}")
            return None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the combine data
        """
        # Replace empty strings with NaN
        df = df.replace('', np.nan)
        
        # Convert numeric columns to float
        numeric_columns = [
            'HEIGHT_W_SHOES', 'HEIGHT_WO_SHOES', 'WEIGHT', 'WINGSPAN',
            'STANDING_REACH', 'BODY_FAT_PCT', 'HAND_LENGTH', 'HAND_WIDTH'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where all numeric columns are NaN
        df = df.dropna(subset=numeric_columns, how='all')
        
        return df

    def get_multiple_years_data(self, start_year: int, num_years: int) -> pd.DataFrame:
        """
        Fetch combine data for multiple years and combine them
        """
        all_data = []
        
        for year in range(start_year, start_year + num_years):
            print(f"Fetching combine data for year {year}...")
            df = self.get_combine_data(year)
            if df is not None:
                all_data.append(df)
            time.sleep(1)  # Add delay to avoid rate limiting
            
        if not all_data:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_data, ignore_index=True)
        return self.preprocess_data(combined_df)

def main():
    # Initialize the NBA combine data fetcher
    nba_combine = NBACombineData()
    
    # Get data for the last 10 combines (2014-2023)
    combined_df = nba_combine.get_multiple_years_data(2014, 10)
    
    if not combined_df.empty:
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(combined_df.info())
        
        print("\nBasic Statistics:")
        print(combined_df.describe())
        
        # Save the processed data to CSV
        combined_df.to_csv('nba_combine_2014_2023.csv', index=False)
        print("\nData saved to 'nba_combine_2014_2023.csv'")
        
        # Display number of players per year
        print("\nNumber of players per draft year:")
        print(combined_df['DRAFT_YEAR'].value_counts().sort_index())

if __name__ == "__main__":
    main() 
    
    