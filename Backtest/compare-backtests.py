#!/usr/bin/env python3
"""
Backtest Comparison Tool

This script allows you to compare results from multiple backtests,
helping you evaluate different parameters and time periods.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from glob import glob
from datetime import datetime
from tabulate import tabulate


def load_backtest_results(results_file):
    """Load backtest results from a JSON file"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Extract basic metadata
        file_name = os.path.basename(results_file)
        name_parts = file_name.replace('_results.json', '').split('_')

        # Try to extract dates from filename
        start_date = name_parts[1] if len(name_parts) > 1 else "Unknown"
        end_date = name_parts[3] if len(name_parts) > 3 else "Unknown"

        # Get key metrics
        metrics = results.get('performance_metrics', {})

        # Create a simple metadata record
        metadata = {
            'file': results_file,
            'name': file_name.replace('_results.json', ''),
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0) * 100,
            'total_pnl': metrics.get('total_pnl', 0),
            'avg_profit_per_trade': metrics.get('avg_profit_per_trade', 0),
            'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'annual_return_pct': metrics.get('annual_return_pct', 0),
            'profit_factor': metrics.get('profit_factor', 0)
        }

        return results, metadata

    except Exception as e:
        print(f"Error loading results from {results_file}: {e}")
        return None, None


def find_backtest_results(results_dir='backtest_results', pattern=None):
    """Find all backtest result files matching a pattern"""
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found")
        return []

    # Build search pattern
    search_pattern = os.path.join(results_dir, 'backtest_*_results.json')
    if pattern:
        search_pattern = os.path.join(results_dir, f'*{pattern}*_results.json')

    # Find matching files
    result_files = glob(search_pattern)
    return sorted(result_files)


def compare_backtests(result_files):
    """Compare multiple backtest results"""
    if not result_files:
        print("No backtest result files found")
        return None

    # Load all results
    all_results = []
    all_metadata = []

    for file in result_files:
        results, metadata = load_backtest_results(file)
        if results and metadata:
            all_results.append(results)
            all_metadata.append(metadata)

    if not all_metadata:
        print("No valid backtest results found")
        return None

    # Convert to DataFrame for easier comparison
    comparison_df = pd.DataFrame(all_metadata)

    return comparison_df, all_results


def plot_equity_curves(comparison_df, all_results):
    """Plot comparative equity curves"""
    if comparison_df is None or not all_results:
        return

    plt.figure(figsize=(12, 8))

    for i, results in enumerate(all_results):
        if 'equity_curve' not in results:
            continue

        # Extract equity curve data
        equity_data = results['equity_curve']
        dates = []
        equity = []

        for point in equity_data:
            if isinstance(point, list) and len(point) == 2:
                date_str, value = point
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    dates.append(date)
                    equity.append(value)
                except:
                    continue

        if dates and equity:
            # Calculate percent change for better comparison
            initial_equity = equity[0]
            equity_pct = [(e / initial_equity - 1) * 100 for e in equity]

            # Plot with a label
            label = comparison_df.iloc[i]['name']
            plt.plot(dates, equity_pct, label=label, linewidth=2)

    plt.title('Equity Curves Comparison (% Change)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Equity Change (%)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    # Save figure
    plt.tight_layout()
    plt.savefig('backtest_results/comparison_equity_curves.png', dpi=300)
    plt.close()

    print(f"Saved equity curves comparison to backtest_results/comparison_equity_curves.png")


def plot_metrics_comparison(comparison_df):
    """Create comparative bar charts of key metrics"""
    if comparison_df is None or len(comparison_df) == 0:
        return

    # Select key metrics for comparison
    metrics = ['win_rate', 'annual_return_pct', 'sharpe_ratio', 'max_drawdown_pct']
    titles = ['Win Rate (%)', 'Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']

    # Create a multi-chart figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Sort by this metric (descending for most, ascending for drawdown)
        ascending = metric == 'max_drawdown_pct'
        sorted_df = comparison_df.sort_values(metric, ascending=ascending)

        # Create bar chart
        ax = axes[i]
        bars = ax.bar(sorted_df['name'], sorted_df[metric])
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('backtest_results/comparison_metrics.png', dpi=300)
    plt.close()

    print(f"Saved metrics comparison to backtest_results/comparison_metrics.png")


def compare_trade_performance(all_results):
    """Compare trade performance across different backtests"""
    if not all_results:
        return

    # Extract trades from all results
    all_trades = []

    for i, results in enumerate(all_results):
        if 'trades' not in results:
            continue

        trades = results['trades']
        backtest_name = f"Backtest {i + 1}"

        for trade in trades:
            trade['backtest'] = backtest_name
            all_trades.append(trade)

    if not all_trades:
        print("No trade data found in results")
        return

    # Convert to DataFrame
    trades_df = pd.DataFrame(all_trades)

    # Analyze by strategy type
    if 'strategy' in trades_df.columns:
        strategy_perf = trades_df.groupby(['backtest', 'strategy'])['total_pnl'].agg(['mean', 'count'])
        print("\n=== STRATEGY PERFORMANCE ACROSS BACKTESTS ===")
        print(tabulate(strategy_perf, headers='keys', tablefmt='pipe', floatfmt='.2f'))

        # Create a bar chart of strategy performance
        plt.figure(figsize=(12, 8))

        # Pivot the data for plotting
        pivot_df = trades_df.pivot_table(
            index='strategy',
            columns='backtest',
            values='total_pnl',
            aggfunc='mean'
        )

        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title('Average P&L by Strategy Across Backtests', fontsize=14)
        plt.xlabel('Strategy Type', fontsize=12)
        plt.ylabel('Average P&L ($)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Backtest")

        plt.tight_layout()
        plt.savefig('backtest_results/comparison_strategy_performance.png', dpi=300)
        plt.close()

        print(f"Saved strategy performance comparison to backtest_results/comparison_strategy_performance.png")

    # Return dataframe for further analysis
    return trades_df


def generate_html_report(comparison_df, output_file='backtest_results/comparison_report.html'):
    """Generate an HTML report of the backtest comparison"""
    if comparison_df is None or len(comparison_df) == 0:
        return

    # Create basic HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333366; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th { background-color: #eeeeff; text-align: left; padding: 8px; }
            td { border: 1px solid #ddd; padding: 8px; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .good { color: green; }
            .bad { color: red; }
            .images { display: flex; flex-direction: column; gap: 20px; margin-top: 20px; }
            img { max-width: 100%; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Earnings Calendar Bot Backtest Comparison</h1>
        <p>Generated on DATETIME</p>

        <h2>Backtest Results Comparison</h2>
        TABLE_PLACEHOLDER

        <h2>Performance Visualizations</h2>
        <div class="images">
            <div>
                <h3>Equity Curves Comparison</h3>
                <img src="comparison_equity_curves.png" alt="Equity Curves Comparison">
            </div>
            <div>
                <h3>Metrics Comparison</h3>
                <img src="comparison_metrics.png" alt="Metrics Comparison">
            </div>
            <div>
                <h3>Strategy Performance</h3>
                <img src="comparison_strategy_performance.png" alt="Strategy Performance">
            </div>
        </div>

        <h2>Analysis and Insights</h2>
        <div id="insights">
            INSIGHTS_PLACEHOLDER
        </div>
    </body>
    </html>
    """

    # Generate insights
    insights = "<ul>"

    # Add insights about best performer
    if not comparison_df.empty:
        # Best annual return
        best_return_idx = comparison_df['annual_return_pct'].idxmax()
        best_return = comparison_df.loc[best_return_idx]
        insights += f"<li>Best annual return: <b>{best_return['annual_return_pct']:.2f}%</b> from {best_return['name']}</li>"

        # Best Sharpe ratio
        best_sharpe_idx = comparison_df['sharpe_ratio'].idxmax()
        best_sharpe = comparison_df.loc[best_sharpe_idx]
        insights += f"<li>Best risk-adjusted return (Sharpe ratio): <b>{best_sharpe['sharpe_ratio']:.2f}</b> from {best_sharpe['name']}</li>"

        # Lowest drawdown
        best_dd_idx = comparison_df['max_drawdown_pct'].idxmin()
        best_dd = comparison_df.loc[best_dd_idx]
        insights += f"<li>Lowest maximum drawdown: <b>{best_dd['max_drawdown_pct']:.2f}%</b> from {best_dd['name']}</li>"

        # Best win rate
        best_wr_idx = comparison_df['win_rate'].idxmax()
        best_wr = comparison_df.loc[best_wr_idx]
        insights += f"<li>Highest win rate: <b>{best_wr['win_rate']:.2f}%</b> from {best_wr['name']}</li>"

        # Largest sample size
        best_sample_idx = comparison_df['total_trades'].idxmax()
        best_sample = comparison_df.loc[best_sample_idx]
        insights += f"<li>Largest sample size: <b>{best_sample['total_trades']}</b> trades from {best_sample['name']}</li>"

    insights += "</ul>"

    # Replace placeholders
    html = html.replace("DATETIME", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("TABLE_PLACEHOLDER", comparison_df.to_html(index=False, float_format=lambda x: f"{x:.2f}"))
    html = html.replace("INSIGHTS_PLACEHOLDER", insights)

    # Save HTML file
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Generated HTML comparison report: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple backtest results')
    parser.add_argument('--dir', type=str, default='backtest_results', help='Directory containing backtest results')
    parser.add_argument('--pattern', type=str, default=None, help='Optional filename pattern to filter results')
    parser.add_argument('--files', nargs='+', help='Specific result files to compare')

    args = parser.parse_args()

    print("=== Backtest Comparison Tool ===")

    # Either use specified files or find files matching pattern
    if args.files:
        result_files = args.files
        print(f"Comparing {len(result_files)} specified result files")
    else:
        result_files = find_backtest_results(args.dir, args.pattern)
        print(f"Found {len(result_files)} result files" +
              (f" matching pattern '{args.pattern}'" if args.pattern else ""))

    # Compare backtests
    comparison_results = compare_backtests(result_files)

    if comparison_results:
        comparison_df, all_results = comparison_results

        # Print comparison table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n=== BACKTEST COMPARISON ===")
        print(tabulate(comparison_df, headers='keys', tablefmt='pipe', floatfmt='.2f'))

        # Create visualizations
        plot_equity_curves(comparison_df, all_results)
        plot_metrics_comparison(comparison_df)
        compare_trade_performance(all_results)

        # Generate HTML report
        generate_html_report(comparison_df)

        # Print summary
        print("\n=== SUMMARY ===")
        best_return_idx = comparison_df['annual_return_pct'].idxmax()
        best_return = comparison_df.loc[best_return_idx]
        best_sharpe_idx = comparison_df['sharpe_ratio'].idxmax()
        best_sharpe = comparison_df.loc[best_sharpe_idx]

        print(f"• Best annual return: {best_return['annual_return_pct']:.2f}% from {best_return['name']}")
        print(
            f"• Best risk-adjusted return: Sharpe ratio of {best_sharpe['sharpe_ratio']:.2f} from {best_sharpe['name']}")

        print("\nDetailed comparison report generated at: backtest_results/comparison_report.html")
    else:
        print("Failed to compare backtests. See errors above.")


if __name__ == "__main__":
    main()