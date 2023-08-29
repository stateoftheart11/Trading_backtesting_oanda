import pandas as pd
import numpy as np
import math
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import pytz
import os
from datetime import datetime
import time

class Strategy:

    def strategy_MACS(self, df, input_parameters):
        ## Moving Average Cross Strategy
        # Extract and preprocess input parameters
        strategy_name = input_parameters["strategy"]
        EMA_big, EMA_medium, EMA_small = input_parameters["EMA_periods"]
        SL = input_parameters["SL"]
        TP = input_parameters["TP"]

        # Add Exponential Moving Averages to DataFrame
        df["EMA_big"] = df["close"].ewm(span=EMA_big).mean()
        df["EMA_medium"] = df["close"].ewm(span=EMA_medium).mean()
        df["EMA_small"] = df["close"].ewm(span=EMA_small).mean()

        # Evaluate trade entry and exit signals
        df["signal"] = np.where(
            (df["EMA_small"] > df["EMA_medium"]) & (df["EMA_medium"] > df["EMA_big"]), 1,
            np.where(
                (df["EMA_small"] < df["EMA_medium"]) & (df["EMA_medium"] < df["EMA_big"]), -1, 0
            )
        )

        # Initialize variables to store trade information and track state
        trades = []
        trade_open = False
        opening_price = 0
        trade_direction = None  # 1 for long trades, -1 for short trades

        for idx, row in df.iterrows():
            # Check if there's an open trade
            if trade_open:
                # Determine if the trade should be closed
                if (
                    (row["signal"] != trade_direction)
                    or (row["close"] - opening_price >= TP * trade_direction)
                    or (row["close"] - opening_price <= (-SL) * trade_direction)
                ):
                    trade_open = False
                    closed_price = row["close"]
                    trades.append((opening_price, closed_price))

            # If no trade is open, check if there's a new signal
            elif row["signal"] != 0:
                trade_open = True
                trade_direction = row["signal"]
                opening_price = row["close"]

        # Compute output values (total return, win rate, average profit, etc.)
        trade_results = []
        for trade in trades:
            trade_results.append(trade[1] - trade[0])

        total_return = sum(trade_results)
        if len(trade_results) != 0:
            win_rate = len([result for result in trade_results if result > 0]) / len(trade_results) * 100
            average_profit = sum(trade_results) / len(trade_results)
        else:
            win_rate = 0
            average_profit = 0
        # Create a DataFrame for trade results
        trade_results_df = pd.DataFrame(trade_results, columns=['returns'])
        
        # Calculate maximum drawdown
        trade_results_df['cum_returns'] = trade_results_df['returns'].cumsum()
        trade_results_df['running_max'] = trade_results_df['cum_returns'].cummax()
        trade_results_df['drawdown'] = trade_results_df['running_max'] - trade_results_df['cum_returns']
        max_drawdown = trade_results_df['drawdown'].max()

        # Calculate risk_reward_ratio
        winning_trades = [result for result in trade_results if result > 0]
        losing_trades = [result for result in trade_results if result < 0]
        
        risk_reward_ratio = abs(sum(winning_trades) / sum(losing_trades))

        # Calculate Sharpe ratio
        avg_daily_return = df['close'].pct_change(1).mean()
        daily_volatility = df['close'].pct_change(1).std()
        sharpe_ratio = avg_daily_return / daily_volatility

        # Add trade_count and exposure_time
        trade_count = len(trades)
        exposure_time = df['signal'].abs().sum()

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "maximum_drawdown": max_drawdown,
            "risk_reward_ratio": risk_reward_ratio,
            "sharpe_ratio": sharpe_ratio,
            "trade_count": trade_count,
            "exposure_time": exposure_time,
        }
    
    def strategy_SimpleMACS(self, df, input_parameters):
        # Extract and preprocess input parameters
        strategy_name = input_parameters["strategy"]
        SMA_fast, SMA_slow = input_parameters["indicator_inputs"]["SMA_periods"]
        SL = input_parameters["SL"]
        TP = input_parameters["TP"]

        # Add Simple Moving Averages to DataFrame
        df["SMA_fast"] = df["close"].rolling(window=SMA_fast).mean()
        df["SMA_slow"] = df["close"].rolling(window=SMA_slow).mean()

        # Evaluate trade entry and exit signals
        df["signal"] = np.where(
            df["SMA_fast"] > df["SMA_slow"], 1,
            np.where(
                df["SMA_fast"] < df["SMA_slow"], -1, 0
            )
        )

        # Initialize variables to store trade information and track state
        trades = []
        trade_open = False
        opening_price = 0
        trade_direction = None
        trade_count = 0

        for idx, row in df.iterrows():
            # Check if there's an open trade
            if trade_open:
                # Determine if the trade should be closed
                if (
                    (row["signal"] != trade_direction)
                    or (row["close"] - opening_price >= TP * trade_direction)
                    or (row["close"] - opening_price <= (-SL) * trade_direction)
                ):
                    trade_open = False
                    closed_price = row["close"]
                    trades.append((opening_price, closed_price))
                    trade_count += 1

            # If no trade is open, check if there's a new signal
            elif row["signal"] != 0:
                trade_open = True
                trade_direction = row["signal"]
                opening_price = row["close"]

        # Compute output values
        trade_results = [trade[1] - trade[0] for trade in trades]
        total_return = sum(trade_results)
        if len(trade_results) != 0:
            win_rate = len([result for result in trade_results if result > 0]) / len(trade_results) * 100
            average_profit = sum(trade_results) / len(trade_results)
        else:
            win_rate = 0
            average_profit = 0

        if trade_results:
            max_drawdown = min(trade_results)
        else:
            max_drawdown = 0
        if sum(result for result in trade_results if result < 0) != 0:
            risk_reward_ratio = abs(sum([result for result in trade_results if result > 0]) / sum([result for result in trade_results if result < 0]))
        else:
            risk_reward_ratio = 0
        sharpe_ratio = total_return / np.std(trade_results)
        exposure_time = trade_count / len(df) * 100

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "maximum_drawdown": max_drawdown,
            "risk_reward_ratio": risk_reward_ratio,
            "sharpe_ratio": sharpe_ratio,
            "trade_count": trade_count,
            "exposure_time": exposure_time,
        }

    def strategy_RSI(self, df, input_parameters):
        # Extract and preprocess input parameters
        strategy_name = input_parameters["strategy"]
        RSI_period = input_parameters["indicator_inputs"]["RSI_period"]
        RSI_overbought = input_parameters["indicator_inputs"]["RSI_overbought"]
        RSI_oversold = input_parameters["indicator_inputs"]["RSI_oversold"]
        SL = input_parameters["SL"]
        TP = input_parameters["TP"]
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=RSI_period).mean()
        avg_loss = loss.rolling(window=RSI_period).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Evaluate trade entry and exit signals
        df["signal"] = np.where(df["RSI"] > RSI_overbought, -1, np.where(df["RSI"] < RSI_oversold, 1, 0))
        
        # Initialize variables to store trade information and track state
        trades = []
        trade_open = False
        opening_price = 0
        trade_direction = None  # 1 for long trades, -1 for short trades
        
        for idx, row in df.iterrows():
            # Check if there's an open trade
            if trade_open:
                # Determine if the trade should be closed
                if (
                    (row["signal"] != trade_direction)
                    or (row["close"] - opening_price >= TP * trade_direction)
                    or (row["close"] - opening_price <= (-SL) * trade_direction)
                ):
                    trade_open = False
                    closed_price = row["close"]
                    trades.append((opening_price, closed_price))

            # If no trade is open, check if there's a new signal
            elif row["signal"] != 0:
                trade_open = True
                trade_direction = row["signal"]
                opening_price = row["close"]
        
        # Compute output values (total return, win rate, average profit, etc.)
        trade_results = []
        for trade in trades:
            trade_results.append(trade[1] - trade[0])

        total_return = sum(trade_results)
        if len(trade_results) != 0:
            win_rate = len([result for result in trade_results if result > 0]) / len(trade_results) * 100
            average_profit = sum(trade_results) / len(trade_results)
        else:
            win_rate = 0
            average_profit = 0
        
        trade_count = len(trades)
        exposure_time = (trade_count * RSI_period / len(df)) * 100
        
        winning_trades = [result for result in trade_results if result > 0]
        losing_trades = [result for result in trade_results if result < 0]
        
        average_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        average_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
        risk_reward_ratio = abs(average_loss / average_win) if average_win != 0 else 0
        
        df["return"] = df["close"].pct_change()
        sharpe_ratio = (df["return"].mean() / df["return"].std()) * np.sqrt(len(df))
        
        df["drawdown"] = df["close"].cummax() - df["close"]
        max_drawdown = df["drawdown"].max()
        
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "average_profit": average_profit,
            "maximum_drawdown": max_drawdown,
            "risk_reward_ratio": risk_reward_ratio,
            "sharpe_ratio": sharpe_ratio,
            "trade_count": trade_count,
            "exposure_time": exposure_time,
        }

    def run_strategy(self, df, input_parameters):
        if input_parameters["strategy"] == "MovingAverageCross":
            return self.strategy_MACS(df, input_parameters)
        
        elif input_parameters["strategy"] == "SimpleMovingAverageCross":
            return self.strategy_SimpleMACS(df, input_parameters)
        
        elif input_parameters["strategy"] == "RSI":
            return self.strategy_RSI(df, input_parameters)