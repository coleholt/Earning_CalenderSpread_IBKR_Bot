#!/usr/bin/env python3
"""
Quick Backtest Runner

This script provides a simplified interface to run a small backtest sample
and generate a quick analysis of the results. Use this to test your setup
before running a full backtest.
"""

import argparse
import datetime
import json
import os
import sys
import time
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Check if the real-world backtester exists
if not os.path.exists('real-world-backtest.py'):
    print("Error: real-world-backtest.py not found. Please make sure the file exists in the current directory.")
    sys.exit(1)


def run_quick_backtest(days_back=30, risk_percent=6.5, symbols_limit=5):
    """Run a quick backtest for a recent period"""
    try:
        # Calculate dates
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days_back)

        # Format dates
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Create a temporary modified config for quick testing
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)

            # Create a test config with specified parameters
            test_config = config.copy()
            test_config['account']['risk_per_trade'] = risk_percent

            # Limit the number of trades per day for faster testing
            if 'max_symbols_per_day' not in test_config:
                test_config['max_symbols_per_day'] = symbols_limit

            # Save temporary test config
            with open('test_config.json', 'w') as f:
                json.dump(test_config, f, indent=2)

            # Build command
            cmd = f"python real-world-backtest.py --start {start_str} --end {end_str} --config test_config.json"

            print(f"Running quick backtest from {start_str} to {end_str} with {risk_percent}% risk per trade")
            print(f"Limiting to max {symbols_limit} symbols per day for faster execution\n")

            # Execute the backtest
            start_time = time.time()
            os.system(cmd)
            elapsed_time = time.time() - start_time

            print(f"\nQuick backtest completed in {elapsed_time:.2f} seconds")

            # Quick results analysis
            result_file = f"backtest_results/backtest_{start_str}_to_{end_str}_results.json"
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    results = json.load(f)

                # Display summary
                print("\n=== QUICK BACKTEST SUMMARY ===")
                metrics = results.get('performance_metrics', {})
                trades = results.get('trades', [])

                if metrics:
                    print(f"Total trades: {metrics.get('total_trades', 0)}")
                    print(f"Win rate: {metrics.get('win_rate', 0) * 100:.2f}%")
                    print(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
                    print(f"Avg. profit per trade: ${metrics.get('avg_profit_per_trade', 0):,.2f}")
                    print(f"Max drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                    print(f"Annual return: {metrics.get('annual_return_pct', 0):.2f}%")

                # Clean up
                os.remove('test_config.json')

                # Return basic results for further analysis
                return results
            else:
                print(f"Error: Results file {result_file} not found")
                return None
        else:
            print("Error: config.json not found. Please make sure your configuration file exists.")
            return None

    except Exception as e:
        print(f"Error running quick backtest: {e}")
        traceback.print_exc()
        return None


def analyze_trade_statistics(results):
    """Analyze trade statistics from backtest results"""
    if not results or 'trades' not in results or not results['trades']:
        print("No trade data available for analysis")
        return

    trades = results['trades']
    df = pd.DataFrame(trades)

    # Convert columns to numeric
    numeric_columns = ['total_pnl', 'entry_debit', 'exit_debit', 'roi_pct', 'days_held']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Trade type analysis
    if 'strategy' in df.columns:
        strategy_counts = df['strategy'].value_counts()
        print("\n=== TRADE TYPE BREAKDOWN ===")
        for strategy, count in strategy_counts.items():
            strategy_df = df[df['strategy'] == strategy]
            win_rate = (strategy_df['total_pnl'] > 0).mean() * 100
            avg_pnl = strategy_df['total_pnl'].mean()
            print(f"{strategy}: {count} trades, {win_rate:.2f}% win rate, ${avg_pnl:.2f} avg P&L")

    # Days held analysis
    if 'days_held' in df.columns:
        avg_days = df['days_held'].mean()
        print(f"\nAverage days held: {avg_days:.2f}")

        # Group by days held
        df['days_group'] = pd.cut(df['days_held'], bins=[0, 1, 2, 3, 5, 10, 100],
                                  labels=['1 day', '2 days', '3 days', '4-5 days', '6-10 days', '10+ days'])
        days_group = df.groupby('days_group')['total_pnl'].agg(['mean', 'count'])
        print("\n=== DAYS HELD ANALYSIS ===")
        print(days_group)

    # Top and bottom performers
    if 'symbol' in df.columns and 'total_pnl' in df.columns:
        top_symbols = df.groupby('symbol')['total_pnl'].sum().sort_values(ascending=False).head(5)
        bottom_symbols = df.groupby('symbol')['total_pnl'].sum().sort_values().head(5)

        print("\n=== TOP PERFORMING SYMBOLS ===")
        for symbol, pnl in top_symbols.items():
            print(f"{symbol}: ${pnl:.2f}")

        print("\n=== WORST PERFORMING SYMBOLS ===")
        for symbol, pnl in bottom_symbols.items():
            print(f"{symbol}: ${pnl:.2f}")

    # Print common patterns
    print("\n=== INSIGHT SUMMARY ===")

    # Check if one strategy outperforms significantly
    if 'strategy' in df.columns:
        strategy_pnl = df.groupby('strategy')['total_pnl'].mean()
        if len(strategy_pnl) > 1:
            best_strategy = strategy_pnl.idxmax()
            worst_strategy = strategy_pnl.idxmin()
            difference = strategy_pnl.max() - strategy_pnl.min()

            if difference > 50:  # $50 threshold for "significant" difference
                print(f"• {best_strategy} seems to outperform {worst_strategy} by ${difference:.2f} per trade")

    # Check for win rate pattern by days held
    if 'days_held' in df.columns and len(df) > 5:
        df['is_win'] = df['total_pnl'] > 0
        days_win_rate = df.groupby('days_group')['is_win'].mean()
        best_days = days_win_rate.idxmax()

        print(f"• Best win rate ({days_win_rate.max() * 100:.2f}%) occurs with {best_days}")

    # Return DataFrame for further analysis
    return df


def main():
    parser = argparse.ArgumentParser(description='Run a quick backtest sample')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest (default: 30)')
    parser.add_argument('--risk', type=float, default=6.5, help='Risk percent per trade (default: 6.5)')
    parser.add_argument('--symbols', type=int, default=5, help='Max symbols per day (default: 5)')

    args = parser.parse_args()

    print("=== Quick Backtest Runner ===")
    results = run_quick_backtest(args.days, args.risk, args.symbols)

    if results:
        trade_df = analyze_trade_statistics(results)

        # Display options for further analysis
        print("\n=== NEXT STEPS ===")
        print("1. Check the 'backtest_results' directory for detailed reports and plots")
        print("2. Run a longer backtest period for more statistical significance")
        print("3. Adjust strategy parameters in config.json to optimize performance")
        print("4. Run with different risk percentages to find optimal risk allocation")
    else:
        print("\nQuick backtest failed. Please check the error messages above.")


if __name__ == "__main__":
    main()