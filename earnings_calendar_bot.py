#!/usr/bin/env python3
"""
Earnings Calendar Bot - Automated Calendar Spread Strategy for Earnings Announcements

This bot implements a calendar spread strategy around earnings announcements,
entering positions shortly before market close and exiting after the next market open.
"""

import finnhub
import json
import datetime
import pandas as pd
import pytz
import time
import sys
import os
import threading  # Explicitly import threading
from croniter import croniter
from ib_insync import *
import math
import logging
import numpy as np
import requests
from scipy import stats
from scipy.interpolate import interp1d
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/earnings_calendar_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Load API keys and configuration
def load_config():
    """Load configuration from config file"""
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


# Initialize configuration
config = load_config()
if config and 'api_keys' in config and 'finnhub' in config['api_keys']:
    FINNHUB_API_KEY = config['api_keys']['finnhub']
else:
    logger.error("Finnhub API key not found in config")
    FINNHUB_API_KEY = ""


# IBKR API configuration
def connect_to_ibkr():
    """Connect to Interactive Brokers TWS or IB Gateway"""
    try:
        ib = IB()

        # Load connection parameters from config
        config = load_config()
        if config and 'ibkr' in config:
            ibkr_config = config['ibkr']
            host = ibkr_config.get('host', '127.0.0.1')
            port = ibkr_config.get('port', 7497)
            client_id = ibkr_config.get('client_id', 1)
        else:
            # Default settings for paper trading
            host = '127.0.0.1'
            port = 7497
            client_id = 1
            logger.warning("Using default IBKR connection settings")

        # Connect to IB
        logger.info(f"Connecting to IBKR at {host}:{port} with client ID {client_id}")
        ib.connect(host, port, clientId=client_id)

        if ib.isConnected():
            logger.info("Successfully connected to IBKR")

            # Verify if using paper trading account
            account_summary = ib.accountSummary()
            account_type = next((item.value for item in account_summary if item.tag == 'AccountType'), None)

            if account_type and 'PAPER' in account_type.upper():
                logger.info("Connected to PAPER trading account")
            elif account_type:
                logger.warning("WARNING: Connected to LIVE trading account!")

            return ib
        else:
            logger.error("Failed to connect to IBKR")
            return None
    except Exception as e:
        logger.error(f"Error connecting to IBKR: {e}")
        return None


# Account size functions
def get_account_size(ib):
    """
    Get the actual account size from IBKR or use the configured fallback value.
    If account size cannot be determined reliably, return 0 to prevent trading.

    Args:
        ib: The IB connection object

    Returns:
        float: The account size (Net Liquidation Value) or 0 if it cannot be determined
    """
    if not ib or not ib.isConnected():
        logger.error("Not connected to IBKR. Cannot get account size. Trading will be prevented.")
        return 0  # Return 0 to prevent trading

    # Load configuration
    config = load_config()
    use_actual_balance = True  # Default to True

    if config and 'account' in config and 'use_actual_balance' in config['account']:
        use_actual_balance = config['account']['use_actual_balance']

    # If configured to use fallback value, don't query IBKR
    if not use_actual_balance:
        logger.info("Using configured account size instead of actual balance")
        return get_fallback_account_size()

    try:
        # Get account summary
        account_summary = ib.accountSummary()

        if not account_summary:
            logger.error("Empty account summary received from IBKR. Trading will be prevented.")
            return 0  # Return 0 to prevent trading

        # Find the Net Liquidation Value
        for item in account_summary:
            if item.tag == 'NetLiquidation':
                try:
                    net_liquidation = float(item.value)
                    # Check for valid account value
                    if net_liquidation <= 0:
                        logger.error(f"Invalid account value: ${net_liquidation:.2f}. Trading will be prevented.")
                        return 0  # Return 0 to prevent trading

                    logger.info(f"Retrieved actual account value: ${net_liquidation:.2f}")
                    return net_liquidation
                except (ValueError, TypeError):
                    logger.error(
                        f"Could not convert NetLiquidation value to float: {item.value}. Trading will be prevented.")
                    return 0  # Return 0 to prevent trading

        # If we get here, we couldn't find Net Liquidation value
        logger.error("NetLiquidation value not found in account summary. Trading will be prevented.")
        return 0  # Return 0 to prevent trading

    except Exception as e:
        logger.error(f"Error getting account size from IBKR: {e}. Trading will be prevented.")
        return 0  # Return 0 to prevent trading


def get_fallback_account_size():
    """
    Get the fallback account size from configuration.
    If the fallback account size cannot be determined reliably, return 0 to prevent trading.

    Returns:
        float: The configured account size or 0 if it cannot be determined
    """
    try:
        config = load_config()
        if config and 'account' in config and 'size' in config['account']:
            account_size = float(config['account']['size'])
            if account_size <= 0:
                logger.error(f"Invalid fallback account size: ${account_size:.2f}. Trading will be prevented.")
                return 0  # Return 0 to prevent trading

            # Check for unrealistically small account sizes
            if account_size < 100:
                logger.warning(
                    f"Unusually small fallback account size: ${account_size:.2f}. Verify your configuration.")

            logger.info(f"Using account size from config: ${account_size:.2f}")
            return account_size
        else:
            logger.error("Account size not configured. Trading will be prevented.")
            return 0  # Return 0 to prevent trading
    except Exception as e:
        logger.error(f"Error getting account size from config: {e}. Trading will be prevented.")
        return 0  # Return 0 to prevent trading


# Email notification settings
def load_email_config():
    """Load email configuration from config file"""
    try:
        config = load_config()
        if not config:
            return None

        email_config = config.get('email', {})

        email_settings = {
            'enabled': email_config.get('enabled', True),
            'smtp_server': email_config.get('smtp_server', 'smtp.gmail.com'),
            'smtp_port': email_config.get('smtp_port', 587),
            'sender_email': email_config.get('sender_email', 'earningsbotcalspread.holt@gmail.com'),
            'sender_password': email_config.get('sender_password', 'Eatmyass1'),
            'recipient_email': email_config.get('recipient_email', 'holt.maven@gmail.com'),
            'notification_level': email_config.get('notification_level', 'ERROR')  # ERROR, INFO, or ALL
        }

        return email_settings
    except Exception as e:
        logger.warning(f"Failed to load email config: {e}")
        # Return default config
        return {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'earningsbotcalspread.holt@gmail.com',
            'sender_password': 'Eatmyass1',
            'recipient_email': 'holt.maven@gmail.com',
            'notification_level': 'ERROR'
        }


def send_email_alert(subject, message, html_content=None):
    """Send email alert"""
    email_config = load_email_config()

    if not email_config or not email_config['enabled'] or not email_config['sender_email'] or not email_config[
        'sender_password']:
        logger.warning("Email notifications disabled or not configured properly")
        return False

    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Earnings Bot Alert: {subject}"
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']

        # Add plain text part
        text_part = MIMEText(message, 'plain')
        msg.attach(text_part)

        # Add HTML part if provided
        if html_content:
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

        # Connect to server
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])

        # Send email
        server.send_message(msg)
        server.quit()

        logger.info(f"Email alert sent: {subject}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return False


def send_error_notification(error_msg, traceback_info=None):
    """Send error notification email"""
    subject = "ERROR - Earnings Bot Malfunction"

    message = f"""
ALERT: Earnings Calendar Bot has encountered an error and may have stopped running.

Error: {error_msg}

Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

    if traceback_info:
        message += f"\nTraceback:\n{traceback_info}"

    message += """
Please check the log file and server status.

This is an automated message from your Earnings Calendar Bot.
"""

    send_email_alert(subject, message)


# Trade history and performance tracking
def initialize_trade_history_db():
    """Initialize or load trade history database"""
    db_file = 'trade_history.json'

    if os.path.exists(db_file):
        try:
            with open(db_file, 'r') as f:
                trade_history = json.load(f)
                logger.info(f"Loaded {len(trade_history.get('trades', []))} historical trades")
                return trade_history
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")

    # Create new trade history database
    trade_history = {
        'trades': [],
        'last_updated': datetime.datetime.now().isoformat()
    }

    save_trade_history(trade_history)
    return trade_history


def save_trade_history(trade_history):
    """Save trade history to database file"""
    try:
        trade_history['last_updated'] = datetime.datetime.now().isoformat()

        with open('trade_history.json', 'w') as f:
            json.dump(trade_history, f, indent=2)

        logger.info(f"Trade history saved with {len(trade_history.get('trades', []))} trades")
        return True
    except Exception as e:
        logger.error(f"Error saving trade history: {e}")
        return False


def add_trade_to_history(trade_info):
    """Add a trade to the history database"""
    try:
        trade_history = initialize_trade_history_db()

        # Convert IB objects to serializable format
        trade_record = {
            'symbol': trade_info['symbol'],
            'strategy': trade_info['strategy'],
            'short_expiry': trade_info['short_expiry'],
            'long_expiry': trade_info['long_expiry'],
            'strike': float(trade_info['strike']),
            'right': trade_info['right'],
            'contracts': trade_info['contracts'],
            'order_id': trade_info['order_id'],
            'entry_time': trade_info['entry_time'],
            'entry_price': float(trade_info['debit']),
            'total_cost': float(trade_info['total_debit']),
            'status': 'open',
            'exit_time': None,
            'exit_price': None,
            'pnl': None,
            'pnl_percent': None
        }

        trade_history['trades'].append(trade_record)
        save_trade_history(trade_history)

        logger.info(f"Added trade for {trade_info['symbol']} to history database")
        return True
    except Exception as e:
        logger.error(f"Error adding trade to history: {e}")
        return False


def update_trade_exit(order_id, exit_price, exit_time=None):
    """Update trade history with exit information"""
    try:
        if not exit_time:
            exit_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        trade_history = initialize_trade_history_db()

        # Find the trade by order ID
        for trade in trade_history['trades']:
            if trade['order_id'] == order_id and trade['status'] == 'open':
                # Calculate P&L
                entry_price = trade['entry_price']
                contracts = trade['contracts']

                pnl = (entry_price - exit_price) * contracts * 100  # 100 shares per contract
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100 if entry_price != 0 else 0

                # Update trade record
                trade['status'] = 'closed'
                trade['exit_time'] = exit_time
                trade['exit_price'] = float(exit_price)
                trade['pnl'] = float(pnl)
                trade['pnl_percent'] = float(pnl_percent)

                save_trade_history(trade_history)
                logger.info(f"Updated trade exit for order {order_id} with P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
                return True

        logger.warning(f"No open trade found with order ID {order_id}")
        return False
    except Exception as e:
        logger.error(f"Error updating trade exit: {e}")
        return False


def generate_performance_report():
    """Generate performance report for different time periods"""
    try:
        trade_history = initialize_trade_history_db()
        trades = trade_history.get('trades', [])

        # Convert string dates to datetime objects for closed trades
        closed_trades = []
        for trade in trades:
            if trade['status'] == 'closed' and trade['exit_time']:
                try:
                    trade['exit_time_dt'] = datetime.datetime.strptime(trade['exit_time'], '%Y-%m-%d %H:%M:%S')
                    closed_trades.append(trade)
                except ValueError:
                    continue

        # Sort trades by exit time
        closed_trades.sort(key=lambda x: x['exit_time_dt'])

        # Get current time for calculating periods
        now = datetime.datetime.now()

        # Define time periods
        periods = {
            'yesterday': now - datetime.timedelta(days=1),
            'last_7_days': now - datetime.timedelta(days=7),
            'last_30_days': now - datetime.timedelta(days=30),
            'last_90_days': now - datetime.timedelta(days=90),
            'last_year': now - datetime.timedelta(days=365),
            'last_2_years': now - datetime.timedelta(days=365 * 2),
            'last_5_years': now - datetime.timedelta(days=365 * 5),
            'all_time': datetime.datetime(1970, 1, 1)  # Beginning of time
        }

        # Calculate performance for each period
        performance = {}
        for period_name, start_date in periods.items():
            period_trades = [t for t in closed_trades if t['exit_time_dt'] >= start_date]

            if not period_trades:
                performance[period_name] = {
                    'trade_count': 0,
                    'total_pnl': 0,
                    'win_count': 0,
                    'loss_count': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'avg_pnl_percent': 0
                }
                continue

            # Calculate metrics
            total_pnl = sum(t['pnl'] for t in period_trades)
            win_trades = [t for t in period_trades if t['pnl'] > 0]
            loss_trades = [t for t in period_trades if t['pnl'] <= 0]

            win_count = len(win_trades)
            loss_count = len(loss_trades)

            win_rate = win_count / len(period_trades) if period_trades else 0

            avg_win = sum(t['pnl'] for t in win_trades) / win_count if win_count > 0 else 0
            avg_loss = sum(t['pnl'] for t in loss_trades) / loss_count if loss_count > 0 else 0

            largest_win = max([t['pnl'] for t in win_trades]) if win_trades else 0
            largest_loss = min([t['pnl'] for t in loss_trades]) if loss_trades else 0

            avg_pnl_percent = sum(t['pnl_percent'] for t in period_trades) / len(period_trades)

            performance[period_name] = {
                'trade_count': len(period_trades),
                'total_pnl': total_pnl,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'avg_pnl_percent': avg_pnl_percent
            }

        # Get open positions
        open_trades = [t for t in trades if t['status'] == 'open']

        return {
            'performance': performance,
            'open_positions': open_trades,
            'total_closed_trades': len(closed_trades),
            'total_open_trades': len(open_trades)
        }

    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return None


def format_performance_report_email(report):
    """Format performance report as email message"""
    if not report:
        return "Error generating performance report", "<p>Error generating performance report</p>"

    # Plain text report
    text_report = "EARNINGS CALENDAR BOT - PERFORMANCE REPORT\n"
    text_report += "=" * 50 + "\n\n"

    # Current open positions
    text_report += f"OPEN POSITIONS ({report['total_open_trades']})\n"
    text_report += "-" * 50 + "\n"

    if report['open_positions']:
        for pos in report['open_positions']:
            text_report += f"{pos['symbol']} - {pos['strategy']} - {pos['contracts']} contracts @ ${pos['entry_price']:.2f}\n"
            text_report += f"  Strike: ${pos['strike']:.2f} {pos['right']} - Short Exp: {pos['short_expiry']} - Long Exp: {pos['long_expiry']}\n"
            text_report += f"  Entry: {pos['entry_time']} - Cost: ${pos['total_cost']:.2f}\n\n"
    else:
        text_report += "No open positions\n\n"

    # Performance periods
    periods_to_show = ['yesterday', 'last_7_days', 'last_30_days', 'last_90_days', 'last_year', 'last_2_years',
                       'last_5_years']
    period_names = {
        'yesterday': 'Yesterday',
        'last_7_days': 'Last 7 Days',
        'last_30_days': 'Last 30 Days',
        'last_90_days': 'Last 90 Days',
        'last_year': 'Last Year',
        'last_2_years': 'Last 2 Years',
        'last_5_years': 'Last 5 Years',
        'all_time': 'All Time'
    }

    text_report += "PERFORMANCE SUMMARY\n"
    text_report += "-" * 50 + "\n"

    for period in periods_to_show:
        perf = report['performance'][period]

        if perf['trade_count'] == 0:
            text_report += f"{period_names[period]}: No trades\n\n"
            continue

        text_report += f"{period_names[period]} ({perf['trade_count']} trades):\n"
        text_report += f"  P&L: ${perf['total_pnl']:.2f} - Avg: {perf['avg_pnl_percent']:.2f}%\n"
        text_report += f"  Win Rate: {perf['win_rate'] * 100:.1f}% ({perf['win_count']} wins, {perf['loss_count']} losses)\n"
        text_report += f"  Avg Win: ${perf['avg_win']:.2f} - Avg Loss: ${perf['avg_loss']:.2f}\n\n"

    # Add all time stats
    all_time = report['performance']['all_time']
    text_report += f"ALL TIME ({all_time['trade_count']} trades):\n"
    text_report += f"  P&L: ${all_time['total_pnl']:.2f} - Avg: {all_time['avg_pnl_percent']:.2f}%\n"
    text_report += f"  Win Rate: {all_time['win_rate'] * 100:.1f}% ({all_time['win_count']} wins, {all_time['loss_count']} losses)\n"

    # HTML version
    html_report = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333366; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th { background-color: #eeeeff; text-align: left; padding: 8px; }
            td { border: 1px solid #ddd; padding: 8px; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .positive { color: green; }
            .negative { color: red; }
            .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Earnings Calendar Bot - Performance Report</h1>

        <h2>Open Positions</h2>
    """

    # Add open positions table
    if report['open_positions']:
        html_report += """
        <table>
            <tr>
                <th>Symbol</th>
                <th>Strategy</th>
                <th>Contracts</th>
                <th>Strike</th>
                <th>Short Exp</th>
                <th>Long Exp</th>
                <th>Entry Date</th>
                <th>Entry Price</th>
                <th>Total Cost</th>
            </tr>
        """

        for pos in report['open_positions']:
            html_report += f"""
            <tr>
                <td>{pos['symbol']}</td>
                <td>{pos['strategy']}</td>
                <td>{pos['contracts']}</td>
                <td>${pos['strike']:.2f} {pos['right']}</td>
                <td>{pos['short_expiry']}</td>
                <td>{pos['long_expiry']}</td>
                <td>{pos['entry_time']}</td>
                <td>${pos['entry_price']:.2f}</td>
                <td>${pos['total_cost']:.2f}</td>
            </tr>
            """

        html_report += "</table>"
    else:
        html_report += "<p>No open positions</p>"

    # Add performance tables for each period
    html_report += "<h2>Performance Summary</h2>"

    for period in periods_to_show:
        perf = report['performance'][period]

        if perf['trade_count'] == 0:
            html_report += f"<h3>{period_names[period]}</h3><p>No trades in this period</p>"
            continue

        pnl_class = "positive" if perf['total_pnl'] > 0 else "negative"

        html_report += f"<h3>{period_names[period]} ({perf['trade_count']} trades)</h3>"
        html_report += f"""
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total P&L</td>
                <td class="{pnl_class}">${perf['total_pnl']:.2f}</td>
            </tr>
            <tr>
                <td>Average P&L %</td>
                <td class="{pnl_class}">{perf['avg_pnl_percent']:.2f}%</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{perf['win_rate'] * 100:.1f}% ({perf['win_count']} wins, {perf['loss_count']} losses)</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td class="positive">${perf['avg_win']:.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td class="negative">${perf['avg_loss']:.2f}</td>
            </tr>
            <tr>
                <td>Largest Win</td>
                <td class="positive">${perf['largest_win']:.2f}</td>
            </tr>
            <tr>
                <td>Largest Loss</td>
                <td class="negative">${perf['largest_loss']:.2f}</td>
            </tr>
        </table>
        """

    # Add all time summary
    all_time = report['performance']['all_time']
    all_time_pnl_class = "positive" if all_time['total_pnl'] > 0 else "negative"

    html_report += f"""
    <div class="summary">
        <h2>All Time Summary ({all_time['trade_count']} trades)</h2>
        <p>Total P&L: <span class="{all_time_pnl_class}">${all_time['total_pnl']:.2f}</span></p>
        <p>Win Rate: {all_time['win_rate'] * 100:.1f}% ({all_time['win_count']} wins, {all_time['loss_count']} losses)</p>
        <p>Average P&L: <span class="{all_time_pnl_class}">{all_time['avg_pnl_percent']:.2f}%</span></p>
    </div>

    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """

    return text_report, html_report


def send_performance_report():
    """Generate and send performance report via email"""
    try:
        report = generate_performance_report()
        text_report, html_report = format_performance_report_email(report)

        subject = "Earnings Calendar Bot - Performance Report"
        send_email_alert(subject, text_report, html_report)

        logger.info("Performance report sent successfully")
        return True
    except Exception as e:
        logger.error(f"Error sending performance report: {e}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return False


def schedule_daily_report():
    """Schedule daily performance report"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)

    # Schedule for 8:00 PM ET
    report_time = now.replace(hour=20, minute=0, second=0, microsecond=0)

    # If report time is in the past, schedule for tomorrow
    if now > report_time:
        report_time = report_time + datetime.timedelta(days=1)

    # Calculate seconds until report time
    seconds_until_report = (report_time - now).total_seconds()

    logger.info(f"Scheduled daily report for {report_time.strftime('%Y-%m-%d %H:%M:%S ET')}")
    logger.info(f"({seconds_until_report:.0f} seconds from now)")

    # Start a timer thread
    def report_timer():
        time.sleep(seconds_until_report)
        send_performance_report()
        # Reschedule for the next day
        schedule_daily_report()

    report_thread = threading.Thread(target=report_timer, daemon=True)
    report_thread.start()


# Add the analysis functions from the example code
def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    """
    Calculate Yang-Zhang volatility estimator (realized volatility)
    """
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc ** 2

    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc ** 2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return float(result.iloc[-1])  # Convert to native Python float
    else:
        return result.dropna()


def build_term_structure(days, ivs):
    """
    Build a term structure curve from days to expiration and implied volatilities
    """
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return float(ivs[0])
        elif dte > days[-1]:
            return float(ivs[-1])
        else:
            return float(spline(dte))

    return term_spline


def setup_watchdog():
    """Set up watchdog to monitor if bot is running and send alerts if it stops"""

    watchdog_file = "watchdog.json"

    # Create or update watchdog file
    def update_watchdog_file():
        try:
            watchdog_data = {
                "last_heartbeat": datetime.datetime.now().isoformat(),
                "status": "running",
                "pid": os.getpid()
            }

            with open(watchdog_file, 'w') as f:
                json.dump(watchdog_data, f)

            logger.debug("Watchdog file updated")
            return True
        except Exception as e:
            logger.error(f"Error updating watchdog file: {e}")
            return False

    # Start watchdog thread
    def watchdog_thread():
        while True:
            try:
                # Update heartbeat every 5 minutes
                update_watchdog_file()
                time.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"Watchdog thread error: {e}")

    # Start the watchdog thread
    thread = threading.Thread(target=watchdog_thread, daemon=True)
    thread.start()

    logger.info("Watchdog monitoring started")


def check_watchdog_status():
    """
    External function to check if bot is running by examining the watchdog file
    This can be called by a separate script or cron job
    """
    watchdog_file = "watchdog.json"

    try:
        if not os.path.exists(watchdog_file):
            logger.error("Watchdog file not found. Bot may not be running.")
            send_error_notification("Watchdog file not found. Bot may have stopped running or never started.")
            return False

        with open(watchdog_file, 'r') as f:
            watchdog_data = json.load(f)

        last_heartbeat = datetime.datetime.fromisoformat(watchdog_data["last_heartbeat"])
        now = datetime.datetime.now()

        # Check if heartbeat is more than 10 minutes old
        if (now - last_heartbeat).total_seconds() > 600:  # 10 minutes
            logger.error(f"Bot heartbeat is stale. Last heartbeat: {last_heartbeat.isoformat()}")

            # Check if process is still running
            pid = watchdog_data.get("pid")
            process_running = False

            if pid:
                try:
                    # Check if process exists (UNIX-specific)
                    os.kill(pid, 0)
                    process_running = True
                except OSError:
                    process_running = False

            if process_running:
                message = f"Bot process (PID {pid}) appears to be running but heartbeat is stale."
            else:
                message = f"Bot process (PID {pid}) is not running. Bot has crashed or been terminated."

            send_error_notification(message)
            return False

        return True

    except Exception as e:
        logger.error(f"Error checking watchdog status: {e}")
        send_error_notification(f"Error checking watchdog status: {e}")
        return False


# Update the analyze_stock_metrics function to fix type warnings
def analyze_stock_metrics(symbol):
    """
    Analyze stock metrics using the provided logic from the example code
    Returns a dict with the three key metrics and whether they meet criteria
    """
    try:
        # Load screening criteria from config
        try:
            with open('config.json', 'r') as config_file:
                config = json.load(config_file)
                criteria = config['strategy']['screening_criteria']
                min_avg_volume = criteria.get('min_avg_volume', 1500000)
                min_iv30_rv30_ratio = criteria.get('min_iv30_rv30_ratio', 1.25)
                max_term_structure_slope = criteria.get('max_term_structure_slope', -0.00406)
        except Exception as e:
            logger.warning(f"Error loading config, using default criteria: {e}")
            # Default criteria
            min_avg_volume = 1500000
            min_iv30_rv30_ratio = 1.25
            max_term_structure_slope = -0.00406

        logger.info(f"Analyzing metrics for {symbol}")

        # Add rate limiting for free tier (5 requests per minute)
        time.sleep(12)  # Sleep for 12 seconds between requests (~5 per minute)

        # Get stock data using yfinance (avoid using Polygon API for basic data)
        stock = yf.Ticker(symbol)

        # Get price history for volatility calculation
        price_history = stock.history(period='3mo')
        if price_history.empty:
            logger.error(f"No price history found for {symbol}")
            return None

        # Calculate 30-day average volume (use yfinance data instead of Polygon)
        avg_volume = float(price_history['Volume'].rolling(30).mean().dropna().iloc[-1])
        logger.info(f"{symbol} 30-day avg volume: {avg_volume:.0f}")

        # Get current price
        current_price = float(price_history['Close'].iloc[-1])

        # Get options chain data for term structure
        if len(stock.options) == 0:
            logger.error(f"No options found for {symbol}")
            return None

        # Get expiration dates
        exp_dates = list(stock.options)
        if len(exp_dates) < 2:
            logger.error(f"Not enough option expirations for {symbol}")
            return None

        # Get ATM IV for each expiration
        atm_iv = {}
        for exp_date in exp_dates:
            try:
                chain = stock.option_chain(exp_date)
                calls = chain.calls
                puts = chain.puts

                if calls.empty or puts.empty:
                    continue

                # Find ATM options - Fix: Use proper pandas DataFrame indexing
                call_diffs = (calls['strike'] - current_price).abs()
                call_idx = call_diffs.idxmin()  # This returns index position
                # Fix: Ensure proper DataFrame access with .loc
                call_iv = float(calls.loc[call_idx, 'impliedVolatility'])

                put_diffs = (puts['strike'] - current_price).abs()
                put_idx = put_diffs.idxmin()
                # Fix: Ensure proper DataFrame access with .loc
                put_iv = float(puts.loc[put_idx, 'impliedVolatility'])

                # Average the call and put IVs
                atm_iv_value = (call_iv + put_iv) / 2.0
                atm_iv[exp_date] = atm_iv_value
            except Exception as e:
                logger.warning(f"Error processing option chain for {exp_date}: {e}")

        if not atm_iv:
            logger.error(f"Could not determine ATM IV for any expiration dates for {symbol}")
            return None

        # Calculate days to expiry for each date
        today = datetime.datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)

        # Build term structure model
        if len(dtes) < 2:
            logger.error(f"Not enough IV data points for term structure for {symbol}")
            return None

        term_spline = build_term_structure(dtes, ivs)

        # Calculate term structure slope (0-45 days)
        min_dte = min(dtes)
        ts_slope_0_45 = (term_spline(45) - term_spline(min_dte)) / (45 - min_dte)
        logger.info(f"{symbol} term structure slope (0-45): {ts_slope_0_45:.6f}")

        # Calculate 30-day RV using Yang-Zhang estimator (use yfinance data)
        rv30 = float(yang_zhang(price_history))

        # Calculate IV/RV ratio
        iv30 = float(term_spline(30))
        iv30_rv30 = iv30 / rv30 if rv30 > 0 else 0
        logger.info(f"{symbol} IV30/RV30 ratio: {iv30_rv30:.2f}")

        # Check if metrics meet criteria
        avg_volume_check = avg_volume >= min_avg_volume
        iv30_rv30_check = iv30_rv30 >= min_iv30_rv30_ratio
        ts_slope_check = ts_slope_0_45 <= max_term_structure_slope  # Negative slope is preferred

        # Return results
        results = {
            'symbol': symbol,
            'avg_volume': float(avg_volume),  # Fix: Explicitly convert pandas values to Python types
            'avg_volume_pass': bool(avg_volume_check),
            'iv30_rv30': float(iv30_rv30),
            'iv30_rv30_pass': bool(iv30_rv30_check),
            'ts_slope_0_45': float(ts_slope_0_45),
            'ts_slope_pass': bool(ts_slope_check),
            'all_criteria_met': bool(avg_volume_check and iv30_rv30_check and ts_slope_check)
        }

        return results

    except Exception as e:
        logger.error(f"Error analyzing metrics for {symbol}: {e}")
        return None


def pre_screen_earnings_symbols():
    """
    Perform early pre-screening of earnings symbols to accommodate API rate limits
    This function runs earlier in the day to find potential candidates
    """
    try:
        logger.info("Starting early pre-screening of earnings symbols")

        # Get today's date and next trading day
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        today = now.date()
        next_trading_day = get_next_trading_day(today)

        # Get earnings calendar for today and next trading day
        earnings_data = get_earnings_calendar(today, next_trading_day)

        # Filter symbols reporting after market close (amc) today
        amc_today = filter_symbols_by_hour(earnings_data, today, 'amc')

        # Filter symbols reporting before market open (bmo) on next trading day
        bmo_next = filter_symbols_by_hour(earnings_data, next_trading_day, 'bmo')

        # Combine symbols (both are potential trades)
        all_symbols = list(set(amc_today + bmo_next))

        logger.info(f"Found {len(all_symbols)} potential earnings plays: {', '.join(all_symbols)}")

        # Save the list of symbols to a file for later use
        with open('earnings_candidates.json', 'w') as f:
            json.dump({
                'date': today.isoformat(),
                'next_trading_day': next_trading_day.isoformat(),
                'amc_today': amc_today,
                'bmo_next': bmo_next,
                'all_symbols': all_symbols,
                'timestamp': datetime.datetime.now().isoformat()
            }, f, indent=2)

        # If we have a reasonable number of symbols, start screening them
        if len(all_symbols) > 0:
            # Pre-screen symbols
            pre_screened_results = {}
            for symbol in all_symbols:
                logger.info(f"Pre-screening {symbol}...")
                metrics = analyze_stock_metrics(symbol)
                if metrics:
                    pre_screened_results[symbol] = metrics
                    logger.info(
                        f"Pre-screening complete for {symbol}: {'PASSED' if metrics.get('all_criteria_met', False) else 'FAILED'}")

            # Save pre-screening results
            with open('pre_screened_symbols.json', 'w') as f:
                # Convert results to serializable format
                serializable_results = {}
                for symbol, result in pre_screened_results.items():
                    serializable_results[symbol] = {
                        'all_criteria_met': bool(result.get('all_criteria_met', False)),
                        'avg_volume': float(result.get('avg_volume', 0)),
                        'avg_volume_pass': bool(result.get('avg_volume_pass', False)),
                        'iv30_rv30': float(result.get('iv30_rv30', 0)),
                        'iv30_rv30_pass': bool(result.get('iv30_rv30_pass', False)),
                        'ts_slope_0_45': float(result.get('ts_slope_0_45', 0)),
                        'ts_slope_pass': bool(result.get('ts_slope_pass', False))
                    }

                json.dump({
                    'date': today.isoformat(),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'results': serializable_results
                }, f, indent=2)

            # Count how many passed screening
            passed_symbols = [s for s, r in pre_screened_results.items() if r.get('all_criteria_met', False)]
            logger.info(f"Pre-screening complete. {len(passed_symbols)} of {len(all_symbols)} symbols passed criteria.")

            return pre_screened_results
        else:
            logger.info("No earnings announcements found for pre-screening")
            return {}

    except Exception as e:
        logger.error(f"Error during pre-screening: {e}")
        traceback_info = traceback.format_exc()
        logger.error(traceback_info)
        send_error_notification(f"Error during pre-screening", traceback_info)
        return {}


def process_earnings_symbols(ib, symbols, account_size=10000):
    """Process earnings symbols and place calendar spread orders"""
    if not ib or not ib.isConnected():
        logger.error("Not connected to IBKR. Cannot process symbols.")
        return []

    # Try to load pre-screened results if available
    pre_screened_results = {}
    try:
        if os.path.exists('pre_screened_symbols.json'):
            with open('pre_screened_symbols.json', 'r') as f:
                pre_screening_data = json.load(f)

                # Check if pre-screening was done today
                pre_screen_date = datetime.date.fromisoformat(pre_screening_data['date'])
                today = datetime.date.today()

                if pre_screen_date == today:
                    pre_screened_results = pre_screening_data.get('results', {})
                    logger.info(f"Loaded {len(pre_screened_results)} pre-screened symbols from earlier today")
                else:
                    logger.info(f"Pre-screening data is from {pre_screen_date}, not using it")
    except Exception as e:
        logger.warning(f"Error loading pre-screened results: {e}")

    # If we have pre-screened results, use them to filter symbols
    screened_symbols = []
    if pre_screened_results:
        for symbol in symbols:
            if symbol in pre_screened_results and pre_screened_results[symbol].get('all_criteria_met', False):
                screened_symbols.append(symbol)
                logger.info(f"Using pre-screening result for {symbol}: PASSED")
            elif symbol in pre_screened_results:
                logger.info(f"Using pre-screening result for {symbol}: FAILED")
    else:
        # No pre-screened results, do screening now
        logger.info("No pre-screening results available, performing screening now")
        symbol_metrics = {}
        for symbol in symbols:
            metrics = analyze_stock_metrics(symbol)
            if metrics and metrics.get('all_criteria_met', False):
                symbol_metrics[symbol] = metrics
                screened_symbols.append(symbol)

    logger.info(f"Screening complete. {len(screened_symbols)} of {len(symbols)} symbols passed screening.")

    active_trades = []

    # Get account size using the dedicated function
    net_liquidation = get_account_size(ib)

    # Safety check: If account size is zero or negative, abort trading
    if net_liquidation <= 0:
        logger.error(f"Invalid account size: ${net_liquidation:.2f}. Trading aborted for safety.")
        send_error_notification(f"Trading aborted: Invalid account size: ${net_liquidation:.2f}")
        return []

    # Get risk percentage from config
    config = load_config()
    risk_percent = 6.5  # Default risk percentage
    if config and 'account' in config and 'risk_per_trade' in config['account']:
        risk_percent = config['account']['risk_per_trade']

    logger.info(f"Account value: ${net_liquidation:.2f}, Risk per position: {risk_percent}%")

    for symbol in screened_symbols:
        logger.info(f"Processing {symbol} for calendar spread")
        option_chain = get_option_chain_for_calendar(ib, symbol)

        if option_chain:
            # Find best calendar spread
            symbol_metrics = pre_screened_results.get(symbol, {}) if pre_screened_results else None
            calendar_spread = find_best_calendar_spread(option_chain, symbol_metrics)

            if calendar_spread:
                logger.info(f"Found {calendar_spread['type']} for {symbol} with debit: ${calendar_spread['debit']:.2f}")

                # Place the order
                trade_info = place_calendar_spread_order(ib, calendar_spread, net_liquidation, risk_percent)

                if trade_info:
                    active_trades.append(trade_info)
                    logger.info(f"Successfully placed order for {symbol}")
                else:
                    logger.error(f"Failed to place order for {symbol}")
            else:
                logger.error(f"No suitable calendar spread found for {symbol}")
        else:
            logger.error(f"Failed to get option chain for {symbol}")

    return active_trades


def find_best_calendar_spread(option_chain, symbol_metrics=None):
    """Find the best calendar spread for the given option chain"""
    if not option_chain or 'options' not in option_chain:
        return None

    symbol = option_chain['symbol']

    # If we have symbol metrics from our screening, use those
    # Otherwise proceed with the spread selection regardless
    if symbol_metrics:
        if not symbol_metrics.get('all_criteria_met', False):
            logger.info(f"Symbol {symbol} did not meet all criteria, skipping")
            return None

    options = option_chain['options']

    # Separate calls and puts
    calls = [opt for opt in options if opt['right'] == 'C']
    puts = [opt for opt in options if opt['right'] == 'P']

    # Separate short and long options
    short_calls = [opt for opt in calls if opt['option_type'] == 'short']
    long_calls = [opt for opt in calls if opt['option_type'] == 'long']
    short_puts = [opt for opt in puts if opt['option_type'] == 'short']
    long_puts = [opt for opt in puts if opt['option_type'] == 'long']

    # Sort by DTE - short as close to 1 DTE, long as close to 30 DTE
    short_calls.sort(key=lambda x: x['dte'])  # Ascending for short
    short_puts.sort(key=lambda x: x['dte'])  # Ascending for short
    long_calls.sort(key=lambda x: abs(x['dte'] - 30))  # Closest to 30 DTE
    long_puts.sort(key=lambda x: abs(x['dte'] - 30))  # Closest to 30 DTE

    best_call_calendar = None
    best_put_calendar = None
    best_call_debit = float('inf')
    best_put_debit = float('inf')

    # Find best call calendar
    for short_call in short_calls:
        for long_call in long_calls:
            # Skip if same expiration
            if short_call['expiration'] == long_call['expiration']:
                continue

            # Calculate debit
            debit = long_call['midpoint'] - short_call['midpoint']

            # If this is the best (lowest debit) call calendar so far, save it
            if debit > 0 and (best_call_calendar is None or debit < best_call_debit):
                best_call_calendar = {
                    'short': short_call,
                    'long': long_call,
                    'debit': debit
                }
                best_call_debit = debit

    # Find best put calendar
    for short_put in short_puts:
        for long_put in long_puts:
            # Skip if same expiration
            if short_put['expiration'] == long_put['expiration']:
                continue

            # Calculate debit
            debit = long_put['midpoint'] - short_put['midpoint']

            # If this is the best (lowest debit) put calendar so far, save it
            if debit > 0 and (best_put_calendar is None or debit < best_put_debit):
                best_put_calendar = {
                    'short': short_put,
                    'long': long_put,
                    'debit': debit
                }
                best_put_debit = debit

    # Choose between call and put calendar based on which has lower debit
    if best_call_calendar and best_put_calendar:
        if best_call_debit <= best_put_debit:
            # Call calendar is better (or equal)
            return {
                'type': 'call_calendar',
                'short': best_call_calendar['short'],
                'long': best_call_calendar['long'],
                'debit': best_call_calendar['debit'],
                'symbol': option_chain['symbol'],
                'underlying_price': option_chain['underlying_price']
            }
        else:
            # Put calendar is better
            return {
                'type': 'put_calendar',
                'short': best_put_calendar['short'],
                'long': best_put_calendar['long'],
                'debit': best_put_calendar['debit'],
                'symbol': option_chain['symbol'],
                'underlying_price': option_chain['underlying_price']
            }
    elif best_call_calendar:
        # Only call calendar available
        return {
            'type': 'call_calendar',
            'short': best_call_calendar['short'],
            'long': best_call_calendar['long'],
            'debit': best_call_calendar['debit'],
            'symbol': option_chain['symbol'],
            'underlying_price': option_chain['underlying_price']
        }
    elif best_put_calendar:
        # Only put calendar available
        return {
            'type': 'put_calendar',
            'short': best_put_calendar['short'],
            'long': best_put_calendar['long'],
            'debit': best_put_calendar['debit'],
            'symbol': option_chain['symbol'],
            'underlying_price': option_chain['underlying_price']
        }
    else:
        # No valid calendar spread found
        return None


def is_market_open():
    """Check if US stock market is open right now using Finnhub's market_status endpoint"""
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

    try:
        market_status = finnhub_client.market_status(exchange="US")
        return market_status.get('isOpen', False)
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return False


def is_market_holiday(date):
    """Check if given date is a market holiday using Finnhub's market_holiday endpoint"""
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

    try:
        # Get all holidays for the current year
        holidays = finnhub_client.market_holiday(exchange="US")

        # Format date as string for comparison
        date_str = date.strftime("%Y-%m-%d")

        # Check if the date is in the holidays list
        for holiday in holidays:
            if holiday.get('date') == date_str:
                return True

        return False
    except Exception as e:
        logger.error(f"Error checking market holidays: {e}")
        return False


def is_trading_day(date):
    """Check if the given date is a trading day (not weekend/holiday)"""
    # Check for weekends
    if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False

    # Check for holidays
    if is_market_holiday(date):
        return False

    return True


def get_next_trading_day(date):
    """Get the next trading day after the given date"""
    next_day = date + datetime.timedelta(days=1)
    while not is_trading_day(next_day):
        next_day = next_day + datetime.timedelta(days=1)
    return next_day


def get_earnings_calendar(from_date, to_date):
    """Get earnings calendar for the specified date range"""
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

    # Format dates as required by API
    from_str = from_date.strftime("%Y-%m-%d")
    to_str = to_date.strftime("%Y-%m-%d")

    try:
        return finnhub_client.earnings_calendar(_from=from_str, to=to_str, symbol="")
    except Exception as e:
        logger.error(f"Error fetching earnings calendar: {e}")
        return {"earningsCalendar": []}


def filter_symbols_by_hour(earnings_data, date, hour):
    """Filter symbols reporting at specific hour on specific date"""
    filtered_symbols = []
    date_str = date.strftime("%Y-%m-%d")

    for item in earnings_data['earningsCalendar']:
        if item['date'] == date_str and item['hour'] == hour:
            filtered_symbols.append(item['symbol'])

    return filtered_symbols


def setup_entry_schedule():
    """Setup timer for entry 15 minutes before market close"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)

    # Market closes at 16:00 ET, so 15:45 ET is 15 minutes before close
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    entry_time = market_close - datetime.timedelta(minutes=15)

    # If entry_time is in the past, schedule for tomorrow
    if now > entry_time:
        tomorrow = now + datetime.timedelta(days=1)
        entry_time = tomorrow.replace(hour=15, minute=45, second=0, microsecond=0)

    # Calculate seconds until entry time
    seconds_until_entry = (entry_time - now).total_seconds()

    logger.info(f"Scheduling entry for {entry_time.strftime('%Y-%m-%d %H:%M:%S ET')}")
    logger.info(f"({seconds_until_entry:.0f} seconds from now)")

    return seconds_until_entry


def monitor_positions_and_exit(ib, active_trades, next_trading_day):
    """Monitor positions for assignment and exit at the scheduled time"""
    eastern = pytz.timezone('US/Eastern')

    # Set up exit schedule
    seconds_until_exit = setup_exit_schedule(next_trading_day)

    # If we're within a buffer of the exit time, just exit now
    if seconds_until_exit < 300:  # Less than 5 minutes
        logger.info("Exit time is very soon, proceeding to exit now")
        exit_positions(ib, active_trades)
        return

    # Otherwise, run a monitoring loop
    check_interval = 300  # Check every 5 minutes
    next_check_time = datetime.datetime.now() + datetime.timedelta(seconds=check_interval)

    while True:
        # Sleep until next check
        sleep_seconds = (next_check_time - datetime.datetime.now()).total_seconds()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        now = datetime.datetime.now(eastern)
        logger.info(f"Checking positions at {now.strftime('%Y-%m-%d %H:%M:%S ET')}")

        # Check if it's exit time
        exit_time = datetime.datetime.now(eastern) + datetime.timedelta(seconds=seconds_until_exit)
        if datetime.datetime.now() >= exit_time:
            logger.info("Exit time reached")
            exit_positions(ib, active_trades)
            break

        # Check for assignment on each position
        for trade_info in list(active_trades):  # Use a copy of the list for safe iteration
            if check_for_assignment(ib, trade_info):
                logger.info(f"Assignment detected for {trade_info['symbol']}, closing position")
                close_position(ib, trade_info)
                # Remove from active trades
                active_trades.remove(trade_info)

        # Update next check time
        next_check_time = datetime.datetime.now() + datetime.timedelta(seconds=check_interval)

        # If no more active trades, break out of loop
        if not active_trades:
            logger.info("No more active trades to monitor")
            break


def exit_positions(ib, active_trades):
    """Exit all positions at market open"""
    for trade_info in list(active_trades):  # Use a copy of the list for safe iteration
        logger.info(f"Exiting position for {trade_info['symbol']}")
        success = close_position(ib, trade_info)
        if success:
            logger.info(f"Successfully closed position for {trade_info['symbol']}")
        else:
            logger.error(f"Failed to close position for {trade_info['symbol']}")


def setup_exit_schedule(next_trading_day):
    """Setup timer for exit 15 minutes after market open on the next trading day"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)

    # Market opens at 9:30 ET, so 9:45 ET is 15 minutes after open
    exit_time = datetime.datetime.combine(
        next_trading_day,
        datetime.time(9, 45, 0)
    )
    exit_time = eastern.localize(exit_time)

    # Calculate seconds until exit time
    seconds_until_exit = (exit_time - now).total_seconds()

    logger.info(f"Scheduling exit for {exit_time.strftime('%Y-%m-%d %H:%M:%S ET')}")
    logger.info(f"({seconds_until_exit:.0f} seconds from now)")

    return seconds_until_exit


def check_for_assignment(ib, trade_info):
    """Check if any options have been assigned"""
    if not ib or not ib.isConnected():
        logger.error("Not connected to IBKR. Cannot check for assignment.")
        return False

    try:
        # Get positions and check if our position is still intact
        positions = ib.positions()
        symbol = trade_info['symbol']

        # Check for stock position (indication of assignment)
        stock_position = next((p for p in positions if p.contract.symbol == symbol and p.contract.secType == 'STK'),
                              None)

        if stock_position:
            logger.warning(f"Detected possible assignment for {symbol} - stock position found")
            return True

        return False

    except Exception as e:
        logger.error(f"Error checking for assignment: {e}")
        return False


def place_calendar_spread_order(ib, calendar_spread, account_size, risk_percent=6.5):
    """Place calendar spread order with trade tracking"""
    if not ib or not ib.isConnected() or not calendar_spread:
        logger.error("Cannot place order: IB not connected or invalid spread")
        return False

    try:
        # Calculate position size based on risk
        max_risk = account_size * (risk_percent / 100)

        # Calculate risk per contract (maximum loss = debit paid)
        risk_per_contract = calendar_spread['debit'] * 100  # Convert to dollars (100 shares per contract)

        # Check if the trade meets our risk parameters
        if risk_per_contract > max_risk:
            logger.warning(
                f"Trade for {calendar_spread['symbol']} exceeds risk tolerance: ${risk_per_contract:.2f} > ${max_risk:.2f}")
            return False

        # Calculate number of contracts
        # We'll use exactly 6.5% of account for each trade
        contracts_to_trade = max(1, int(max_risk / risk_per_contract))

        short_contract = calendar_spread['short']['contract']
        long_contract = calendar_spread['long']['contract']

        logger.info(f"Placing {calendar_spread['type']} calendar spread for {calendar_spread['symbol']}")
        logger.info(
            f"Short leg: {short_contract.lastTradeDateOrContractMonth} {short_contract.strike} {short_contract.right}")
        logger.info(
            f"Long leg: {long_contract.lastTradeDateOrContractMonth} {long_contract.strike} {long_contract.right}")
        logger.info(f"Debit: ${calendar_spread['debit']:.2f} per contract")
        logger.info(
            f"Trading {contracts_to_trade} contracts with total risk ${risk_per_contract * contracts_to_trade:.2f}")

        # Create a bag contract for the calendar spread
        calendar = Contract()
        calendar.symbol = short_contract.symbol
        calendar.secType = 'BAG'
        calendar.currency = 'USD'
        calendar.exchange = 'SMART'

        # Define the legs
        leg1 = ComboLeg()
        leg1.conId = short_contract.conId  # Short leg
        leg1.ratio = 1
        leg1.action = 'SELL'
        leg1.exchange = 'SMART'

        leg2 = ComboLeg()
        leg2.conId = long_contract.conId  # Long leg
        leg2.ratio = 1
        leg2.action = 'BUY'
        leg2.exchange = 'SMART'

        calendar.comboLegs = [leg1, leg2]

        # Place the order with limit price slightly above the debit
        # Add small buffer to increase chance of fill
        limit_price = calendar_spread['debit'] * 1.05
        order = LimitOrder('BUY', contracts_to_trade, limit_price)

        # Add order properties for spread orders
        order.algoStrategy = ''
        order.algoParams = []
        order.smartComboRoutingParams = [
            TagValue('NonGuaranteed', '1'),
        ]

        # Submit the order
        trade = ib.placeOrder(calendar, order)
        logger.info(f"Placed calendar spread order: {trade}")

        # Save order details for later reference
        trade_info = {
            'symbol': calendar_spread['symbol'],
            'strategy': calendar_spread['type'],
            'short_expiry': short_contract.lastTradeDateOrContractMonth,
            'long_expiry': long_contract.lastTradeDateOrContractMonth,
            'strike': short_contract.strike,
            'right': short_contract.right,
            'contracts': contracts_to_trade,
            'order_id': trade.order.orderId,
            'entry_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'debit': calendar_spread['debit'],
            'total_debit': calendar_spread['debit'] * contracts_to_trade * 100,
            'trade': trade
        }

        # Add the trade to our history database
        add_trade_to_history(trade_info)

        return trade_info

    except Exception as e:
        logger.error(f"Error placing calendar spread order: {e}")
        return False


def close_position(ib, trade_info):
    """Close an existing position and record the exit"""
    if not ib or not ib.isConnected():
        logger.error("Not connected to IBKR. Cannot close position.")
        return False

    try:
        # Retrieve the order and contract from trade_info
        order_id = trade_info['order_id']

        # Get all open trades
        open_trades = ib.openTrades()

        # Find our specific trade
        trade = next((t for t in open_trades if t.order.orderId == order_id), None)

        if not trade:
            logger.error(f"Could not find trade with order ID {order_id}")
            return False

        # Create a closing order (reverse of the original order)
        close_order = MarketOrder('SELL', trade_info['contracts'])

        # Submit the closing order using the same contract
        close_trade = ib.placeOrder(trade.contract, close_order)

        # Get the exit price - wait for fill if needed
        exit_price = None
        max_attempts = 10
        attempt = 0

        while exit_price is None and attempt < max_attempts:
            ib.sleep(1)  # Wait for order to start executing
            fills = ib.fills()

            # Find our close order in fills
            for fill in fills:
                if fill.execution.orderId == close_trade.order.orderId:
                    exit_price = fill.execution.price
                    break

            attempt += 1

        # If we couldn't get the actual fill price, use current market price as estimate
        if exit_price is None:
            ticker = ib.reqMktData(trade.contract)
            ib.sleep(1)
            exit_price = ticker.marketPrice()
            ib.cancelMktData(trade.contract)

        # Record trade exit in history
        if exit_price:
            update_trade_exit(order_id, exit_price)

        logger.info(f"Placed closing order for {trade_info['symbol']} {trade_info['strategy']} at price {exit_price}")
        return True

    except Exception as e:
        logger.error(f"Error closing position: {e}")
        traceback_info = traceback.format_exc()
        logger.error(traceback_info)
        send_error_notification(f"Error closing position for {trade_info['symbol']}", traceback_info)
        return False


def calculate_dte(expiration_date):
    """Calculate trading days to expiration from today

    Returns trading days to expiration, with today being 0 DTE if expiring today.
    Uses only trading days (excludes weekends and market holidays).
    """
    today = datetime.date.today()
    expiry = datetime.datetime.strptime(expiration_date, '%Y%m%d').date()

    # If expiring today, return 0
    if expiry == today:
        return 0

    # If expiry is before today (shouldn't happen, but just in case)
    if expiry < today:
        return 0

    # Count trading days between today and expiry
    trading_days = 0
    current_day = today

    while current_day < expiry:
        current_day += datetime.timedelta(days=1)
        if is_trading_day(current_day):
            trading_days += 1

    return trading_days


def get_option_chain_for_calendar(ib, symbol):
    """Get relevant option chains for calendar spread strategy

    For calendar spreads around earnings, we need:
    - Short leg: 1-7 trading DTE (must be at least 1 DTE, meaning it expires on or after exit day)
    - Long leg: 21-45 trading DTE (ideally around 30 trading DTE)
    """
    if not ib or not ib.isConnected():
        logger.error("Not connected to IBKR")
        return None

    try:
        # Create a contract for the underlying
        contract = Stock(symbol, 'SMART', 'USD')

        # Qualify the contract
        ib.qualifyContracts(contract)

        # Request market data to get current price
        ticker = ib.reqMktData(contract)
        ib.sleep(1)  # Give time for market data to arrive
        current_price = ticker.marketPrice()
        if math.isnan(current_price) or current_price <= 0:
            # Try last price if market price not available
            current_price = ticker.last
            if math.isnan(current_price) or current_price <= 0:
                # Try close price if last price not available
                current_price = ticker.close
                if math.isnan(current_price) or current_price <= 0:
                    logger.error(f"Could not determine price for {symbol}")
                    return None

        logger.info(f"{symbol} current price: ${current_price:.2f}")

        # Cancel market data subscription
        ib.cancelMktData(contract)

        # Request option chains
        chains = ib.reqSecDefOptParams(contract.symbol, '', contract.secType, contract.conId)

        if not chains:
            logger.error(f"No option chains found for {symbol}")
            return None

        # Filter for options on the SMART exchange
        chain = next((c for c in chains if c.exchange == 'SMART'), None)
        if not chain:
            logger.error(f"No SMART exchange options found for {symbol}")
            return None

        # Find the strike closest to current price
        strikes = sorted(chain.strikes)
        atm_strike = min(strikes, key=lambda x: abs(x - current_price))
        logger.info(f"ATM strike selected: ${atm_strike:.2f}")

        # Get all expirations and calculate trading DTE
        expiration_dtes = {}
        for exp in chain.expirations:
            dte = calculate_dte(exp)
            expiration_dtes[exp] = dte
            logger.debug(f"Expiration {exp} has {dte} trading days to expiry")

        # Filter expirations based on our strategy parameters
        # Short leg: 1-7 trading DTE (must be at least 1 DTE - expiring on exit day or later)
        short_expirations = [exp for exp, dte in expiration_dtes.items() if 1 <= dte <= 7]
        short_expirations.sort(key=lambda x: expiration_dtes[x])  # Sort by DTE ascending

        # Long leg: 21-45 trading DTE (as close to 30 DTE as possible)
        long_expirations = [exp for exp, dte in expiration_dtes.items() if 21 <= dte <= 45]
        long_expirations.sort(key=lambda x: abs(expiration_dtes[x] - 30))  # Sort by closeness to 30 DTE

        if not short_expirations:
            logger.error(f"No suitable short-term expirations (1-7 DTE) found for {symbol}")
            return None

        if not long_expirations:
            logger.error(f"No suitable long-term expirations (21-45 DTE) found for {symbol}")
            return None

        # Create option contracts for calls and puts at ATM strike
        call_options = []
        put_options = []

        # Add short-term options (for selling)
        for exp in short_expirations:
            # Create call option
            call = Option(symbol, exp, atm_strike, 'C', 'SMART')
            call_options.append((call, 'short', expiration_dtes[exp]))

            # Create put option
            put = Option(symbol, exp, atm_strike, 'P', 'SMART')
            put_options.append((put, 'short', expiration_dtes[exp]))

        # Add long-term options (for buying)
        for exp in long_expirations:
            # Create call option
            call = Option(symbol, exp, atm_strike, 'C', 'SMART')
            call_options.append((call, 'long', expiration_dtes[exp]))

            # Create put option
            put = Option(symbol, exp, atm_strike, 'P', 'SMART')
            put_options.append((put, 'long', expiration_dtes[exp]))

        # Extract all contracts for qualification
        all_contracts = [contract[0] for contract in call_options + put_options]

        # Qualify all contracts at once
        ib.qualifyContracts(*all_contracts)

        # Request market data for all options
        option_data = []

        # Process in smaller batches to avoid overwhelming the API
        batch_size = 50
        for i in range(0, len(all_contracts), batch_size):
            batch = all_contracts[i:i + batch_size]

            # Request market data
            tickers = ib.reqTickers(*batch)

            # Process the data - match tickers with contracts and their attributes
            for ticker, (contract, option_type, dte) in zip(tickers,
                                                            call_options + put_options if i == 0
                                                            else (call_options + put_options)[i:i + batch_size]):
                if ticker.contract.right == 'C':
                    option_kind = 'call'
                else:
                    option_kind = 'put'

                # Calculate midprice
                midprice = (ticker.ask + ticker.bid) / 2 if ticker.ask > 0 and ticker.bid > 0 else 0

                option_info = {
                    'contract': contract,
                    'expiration': contract.lastTradeDateOrContractMonth,
                    'strike': contract.strike,
                    'right': contract.right,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'midpoint': midprice,
                    'option_type': option_type,  # 'short' or 'long'
                    'dte': dte,
                    'kind': option_kind,  # 'call' or 'put'
                    'iv': ticker.impliedVol  # Capture implied volatility
                }
                option_data.append(option_info)

        # Calculate term structure slope
        term_structure = calculate_term_structure(option_data)

        return {
            'symbol': symbol,
            'underlying_price': current_price,
            'atm_strike': atm_strike,
            'options': option_data,
            'term_structure': term_structure
        }

    except Exception as e:
        logger.error(f"Error getting option chain for {symbol}: {e}")
        return None


def calculate_term_structure(option_data):
    """Calculate the term structure slope from option data"""
    # Extract calls at the same strike, different expirations
    calls = [opt for opt in option_data if opt['right'] == 'C']

    # Group by strike
    strikes = {}
    for call in calls:
        strike = call['strike']
        if strike not in strikes:
            strikes[strike] = []
        strikes[strike].append(call)

    # Find strike with most expirations
    if not strikes:
        return 0

    best_strike = max(strikes.keys(), key=lambda s: len(strikes[s]))
    expirations = strikes[best_strike]

    if len(expirations) < 2:
        return 0  # Not enough data

    # Sort by DTE
    expirations.sort(key=lambda x: x['dte'])

    # Calculate slope using linear regression
    dtes = [exp['dte'] for exp in expirations]
    ivs = [exp['iv'] for exp in expirations]

    if len(dtes) < 2 or any(math.isnan(iv) for iv in ivs):
        return 0

    slope, _, _, _, _ = stats.linregress(dtes, ivs)
    return float(slope)  # Explicitly convert to Python float


def main():
    """Main function to run the earnings calendar bot with error handling and reporting"""
    try:
        # Set up watchdog monitoring
        setup_watchdog()

        # Schedule daily performance report
        schedule_daily_report()

        # Get today's date in US Eastern time (market timezone)
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        today = now.date()

        # Get next trading day for exit planning
        next_trading_day = get_next_trading_day(today)

        # Check market status
        market_open = is_market_open()
        logger.info(f"Market is currently {'OPEN' if market_open else 'CLOSED'}")

        # Check if today is a trading day
        today_is_trading_day = is_trading_day(today)

        logger.info(f"Today ({today}): Trading day: {'Yes' if today_is_trading_day else 'No'}")
        logger.info(f"Next trading day: {next_trading_day}")

        # Check if we're running in pre-screening mode
        if len(sys.argv) > 1 and sys.argv[1] == "prescreen":
            if today_is_trading_day:
                logger.info("Running in pre-screening mode")
                pre_screen_earnings_symbols()
                logger.info("Pre-screening complete")
            else:
                logger.info("Today is not a trading day. Skipping pre-screening.")
            return True

        # If today is a trading day, proceed with the bot
        if today_is_trading_day:
            # Check if we need to run early pre-screening first
            morning_hours = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now < morning_hours:
                logger.info("Early morning detected, running pre-screening before market open")
                pre_screen_earnings_symbols()

            # Get earnings calendar for today and next trading day
            earnings_data = get_earnings_calendar(today, next_trading_day)

            # Filter symbols reporting after market close (amc) today
            amc_today = filter_symbols_by_hour(earnings_data, today, 'amc')

            # Filter symbols reporting before market open (bmo) on next trading day
            bmo_next = filter_symbols_by_hour(earnings_data, next_trading_day, 'bmo')

            # Combine symbols (both are potential trades)
            all_symbols = list(set(amc_today + bmo_next))

            logger.info(f"Found {len(all_symbols)} potential earnings plays: {', '.join(all_symbols)}")

            if all_symbols:
                # Connect to IBKR
                ib = connect_to_ibkr()

                if ib and ib.isConnected():
                    # For testing/manual mode, process symbols immediately
                    if len(sys.argv) > 1 and sys.argv[1] == "run":
                        logger.info("Running in immediate mode")
                        active_trades = process_earnings_symbols(ib, all_symbols)

                        if active_trades:
                            logger.info(f"Successfully placed {len(active_trades)} trades")
                            # Monitor and exit at appropriate time
                            monitor_positions_and_exit(ib, active_trades, next_trading_day)
                        else:
                            logger.warning("No trades were placed")

                    # Normal mode - schedule entry for 15 minutes before close
                    else:
                        seconds_until_entry = setup_entry_schedule()

                        if seconds_until_entry > 0:
                            logger.info(f"Waiting {seconds_until_entry:.0f} seconds until entry time")
                            time.sleep(seconds_until_entry)

                            # Time to enter positions
                            logger.info("Entry time reached, processing symbols")
                            active_trades = process_earnings_symbols(ib, all_symbols)

                            if active_trades:
                                logger.info(f"Successfully placed {len(active_trades)} trades")
                                # Monitor and exit at appropriate time
                                monitor_positions_and_exit(ib, active_trades, next_trading_day)
                            else:
                                logger.warning("No trades were placed")

                    # Disconnect from IBKR
                    ib.disconnect()
                else:
                    error_msg = "Failed to connect to IBKR"
                    logger.error(error_msg)
                    send_error_notification(error_msg)
            else:
                logger.info("No earnings announcements found for the specified time periods")
        else:
            logger.info("Today is not a trading day. Exiting.")

        # Send daily performance report if not already scheduled
        send_performance_report()

        return True

    except Exception as e:
        error_msg = f"Unexpected error in main function: {e}"
        logger.error(error_msg)
        traceback_info = traceback.format_exc()
        logger.error(traceback_info)
        send_error_notification(error_msg, traceback_info)
        return False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot manually interrupted")
    except Exception as e:
        error_msg = f"Critical error: {e}"
        logger.error(error_msg)
        traceback_info = traceback.format_exc()
        logger.error(traceback_info)
        send_error_notification(error_msg, traceback_info)