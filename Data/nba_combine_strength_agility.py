import pandas as pd
import requests
import time

class NBACombineStrengthAgility:
    def __init__(self):
        self.api_url = "https://stats.nba.com/stats/draftcombinestats"
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.nba.com/',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }

    def get_strength_agility_data(self, year: int) -> pd.DataFrame:
        print(f"Fetching data for {year}...")
        params = {
            'LeagueID': '00',
            'SeasonYear': f"{year}"
        }

        response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract table headers and rows
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']

        df = pd.DataFrame(rows, columns=headers)
        df['DRAFT_YEAR'] = year
        return df

    def get_multiple_years_data(self, start_year: int, num_years: int) -> pd.DataFrame:
        all_data = []
        for year in range(start_year, start_year + num_years):
            try:
                df = self.get_strength_agility_data(year)
                if not df.empty:
                    all_data.append(df)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"Failed for {year}: {e}")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def main():
    nba = NBACombineStrengthAgility()

    # Fetch data from 2014 to 2023 (10 years)
    df = nba.get_multiple_years_data(2014, 10)

    if not df.empty:
        # Save to CSV
        df.to_csv('nba_combine_strength_agility_2014_2023.csv', index=False)
        print("\nSaved data to 'nba_combine_strength_agility_2014_2023.csv'")
        print(df.head())
    else:
        print("No data retrieved.")


if __name__ == "__main__":
    main()
