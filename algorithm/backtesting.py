import pandas as pd
import numpy as np
import math
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import pytz
import os
from datetime import datetime
import time
from dotenv import load_dotenv
from algorithm.strategy import Strategy

class Backtesting:
    SAMPLES_INTERVAL_DURATION = 86400  # You can set it to a value that fits your needs
    API_REQUEST_INTERVAL = 1  # Adjust as needed (e.g., 1 second)

    def __init__(self, api_key):
        self.api_key = api_key
        self.api = API(access_token=self.api_key)
    
    def _historical_data_filename(self, instrument, timeframe, start_date, end_date):
        date_format = "%m_%d_%Y"
        start = start_date.strftime(date_format)
        end = end_date.strftime(date_format)
        return f"{instrument}_{timeframe}_{start}_TO_{end}.csv"

    def save_historical_data_to_csv(self, df, instrument, start_date, end_date, timeframe):
        filename = self._historical_data_filename(instrument, timeframe, start_date, end_date)
        df.to_csv(filename, index=False)

    def load_historical_data_from_csv(self, instrument, timeframe, start_date, end_date):
        filename = self._historical_data_filename(instrument, timeframe, start_date, end_date)
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return None
    
    def parse_historical_data(self, raw_data):
        # Define the column names for the parsed data
        column_names = [
            'time', 'open', 'high', 'low', 'close', 'volume'
        ]

        # Initialize an empty list to store the parsed data
        parsed_data = []

        # Iterate through the raw_data and extract data for each candle
        for candle in raw_data:
            time = candle['time']
            mid = candle['mid']
            volume = candle['volume']

            # Append the extracted data to the parsed_data list as a tuple
            parsed_data.append((
                time,
                float(mid['o']), float(mid['h']), float(mid['l']), float(mid['c']),
                int(volume)
            ))

        # Convert the parsed_data list into a pandas DataFrame
        dataframe = pd.DataFrame(parsed_data, columns=column_names)

        # Return the parsed data as a DataFrame
        return dataframe
    
    def fetch_historical_data(self, instrument, start_date, end_date, timeframe):
        # Convert dates from string to datetime objects
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        directory = f"data_csv/{instrument}/{timeframe}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filename = f"{instrument}-{timeframe}-{start_date}-{end_date}.csv"
        file_path = f"{directory}/{filename}"
        
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col="time", parse_dates=True)
        else:
            start_seconds = int(start_date.timestamp())
            end_seconds = int(end_date.timestamp())
            
            data = []
            while start_seconds < end_seconds:
                end_interval = int(min(end_seconds, start_seconds + self.SAMPLES_INTERVAL_DURATION))
                start_time = time.time()
                try:
                    history = instruments.InstrumentsCandles(
                        instrument=instrument,
                        params={
                            'granularity': timeframe,
                            'from': start_seconds,
                            'to': end_interval,
                        },
                    )
                    self.api.request(history)
                    data += history.response['candles']
                except Exception as e:
                    print(e)
                    time.sleep(60 - (time.time() - start_time))
                    continue
                
                start_seconds = end_interval
                time.sleep(max(self.API_REQUEST_INTERVAL - (time.time() - start_time), 0))
            
            if len(data) > 0:
                data = self.parse_historical_data(data)
                data.to_csv(file_path)
        # Add this line before returning data
        # print("Fetched historical data:", data)
        return data

    def preprocess_data(self, historical_data: pd.DataFrame):
        processed_data = []

        for index, row in historical_data.iterrows():
            candle = {
                "time": index,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            processed_data.append(candle)

        # If you want to remove specific columns, uncomment this part and sspecify the columns.
        # historical_data.drop(["Unnamed: 0"], axis=1, inplace=True)
        
        return processed_data

    def run_backtest(self, instrument, start_date, end_date, timeframe, input_parameters, trading_hours=None):
        # Fetch historical data and preprocess it
        historical_data = self.fetch_historical_data(
            instrument=instrument,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
        )

        if trading_hours is not None:
            start_hour, end_hour = trading_hours
            historical_data = historical_data[(historical_data.index.hour >= start_hour) & (historical_data.index.hour < end_hour)]

        processed_data = self.preprocess_data(historical_data)

        # Convert the list of dictionaries back to a DataFrame
        df = pd.DataFrame(processed_data)

        # Run the strategy on the processed data
        results = Strategy().run_strategy(df, input_parameters)

        return results
    def test_hardcoding(self):
        # Define your input parameters (Timeframe, currency pair, etc.)
        request_backtesting = {
            "input_parameters": {
                "strategy": "SimpleMovingAverageCross",
                "currency_pair_groups": "JPY_Group",
                "SL": 0.01,
                "TP": 0.02,
                "EMA_periods": [200, 50, 20],
                "indicator_inputs": {"SMA_periods": [5, 20]},
                "order_settings_template": "lot_1_20_10"
            },
            "instrument": "EUR_USD",
            "start_date": "2020-01-01",
            "end_date": "2020-01-04",
            "timeframe": "H2",
            "trading_hours": [9, 16]
        }
        print(request_backtesting)

        results = self.run_backtest(request_backtesting["instrument"], request_backtesting["start_date"], request_backtesting["end_date"], request_backtesting["timeframe"], request_backtesting["input_parameters"], request_backtesting["trading_hours"])

        # Analyze and print backtesting results
        print("Backtesting results for {} from {} to {}:".format(request_backtesting["instrument"], request_backtesting["start_date"], request_backtesting["end_date"]))
        print("-" * 40)
        print("Total Return: {:.2f}".format(results["total_return"]))
        print("Win Rate: {:.2f}%".format(results["win_rate"]))
        print("Average Profit per Trade: {:.5f}".format(results["average_profit"]))
        print("Maximum Drawdown: {:.2f}".format(results["maximum_drawdown"]))
        print("Risk/Reward Ratio: {:.2f}".format(results["risk_reward_ratio"]))
        print("Sharpe Ratio: {:.2f}".format(results["sharpe_ratio"]))
        print("Trade Count: {}".format(results["trade_count"]))
        print("Exposure Time: {:.2f}%".format(results["exposure_time"]))
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.environ.get('API_KEY')
    backtesting = Backtesting(api_key)
    backtesting.test_hardcoding()