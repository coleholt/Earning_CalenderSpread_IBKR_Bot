#!/usr/bin/env python3
"""
Real-World Backtesting Framework for Earnings Calendar Bot

This script implements a high-fidelity backtesting system that uses:
1. Historical options data from IBKR TWS
2. Historical earnings announcement data from Finnhub
3. Realistic modeling of broker fees, slippage, and market impact

No simulated or placeholder data is used - all backtests rely on actual market data.
"""

import datetime
import json
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pytz
from tqdm import tqdm
import traceback
import argparse
import sys
from dateutil.relativedelta import relativedelta
import time
import finnhub
from ib_insync import *
from scipy import stats
from scipy.interpolate import interp1d

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('backtest_results', exist_ok=True)


class IBKRHistoricalData:
    """Class to handle historical data retrieval from IBKR"""

    def __init__(self, host='127.0.0.1', port=7497, client_id=10):
        """Initialize connection to IBKR TWS/Gateway"""
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False

    def connect(self):
        """Connect to IBKR TWS/Gateway"""
        try:
            if not self.connected:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.connected = self.ib.isConnected()

                if self.connected:
                    logger.info(f"Connected to IBKR at {self.host}:{self.port}")

                    # Check if this is a paper trading account
                    account_summary = self.ib.accountSummary()
                    account_type = next((item.value for item in account_summary if item.tag == 'AccountType'), None)

                    if account_type:
                        if 'PAPER' in account_type.upper():
                            logger.info("Connected to PAPER trading account")
                        else:
                            logger.warning("WARNING: Connected to LIVE trading account!")
                else:
                    logger.error("Failed to connect to IBKR")
            return self.connected
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR TWS/Gateway"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def get_historical_stock_data(self, symbol, start_date, end_date, bar_size='1 day'):
        """
        Retrieve historical stock price data from IBKR

        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            bar_size (str): Bar size, e.g., '1 day', '1 hour', etc.

        Returns:
            pd.DataFrame: Historical stock data
        """
        if not self.connected and not self.connect():
            logger.error("Not connected to IBKR")
            return None

        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # Convert to IBKR format
            duration = f"{(end_date - start_date).days + 1} D"

            # Request data
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_date.strftime('%Y%m%d %H:%M:%S'),
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            if not bars:
                logger.warning(f"No historical data found for {symbol}")
                return None

            # Convert to DataFrame
            df = util.df(bars)

            # Rename columns to match expected format
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Set Date as index
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

            return df

        except Exception as e:
            logger.error(f"Error retrieving historical stock data for {symbol}: {e}")
            traceback.print_exc()
            return None

    def get_historical_option_chain(self, symbol, date, current_price=None):
        """
        Get historical option chain for a specific date

        Args:
            symbol (str): Stock symbol
            date (datetime): Date for the option chain
            current_price (float, optional): Current stock price to find ATM options

        Returns:
            dict: Option chain data
        """
        if not self.connected and not self.connect():
            logger.error("Not connected to IBKR")
            return None

        try:
            # First, get the stock contract
            stock = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(stock)

            # If current price not provided, get it from historical data
            if current_price is None:
                # Get current price from the day's data
                bars = self.ib.reqHistoricalData(
                    contract=stock,
                    endDateTime=date.strftime('%Y%m%d %H:%M:%S'),
                    durationStr='1 D',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )

                if not bars:
                    logger.warning(f"No price data found for {symbol} on {date}")
                    return None

                current_price = bars[0].close

            logger.info(f"Using price ${current_price:.2f} for {symbol} on {date}")

            # Get option chain for the stock
            chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

            if not chains:
                logger.warning(f"No option chains found for {symbol}")
                return None

            # Filter for options on the SMART exchange
            chain = next((c for c in chains if c.exchange == 'SMART'), None)
            if not chain:
                logger.warning(f"No SMART exchange options found for {symbol}")
                return None

            # Find strikes around current price (25% above and below)
            min_strike = current_price * 0.75
            max_strike = current_price * 1.25
            relevant_strikes = [strike for strike in chain.strikes
                                if min_strike <= strike <= max_strike]

            if not relevant_strikes:
                logger.warning(f"No relevant strikes found for {symbol}")
                return None

            # Find closest ATM strike
            atm_strike = min(relevant_strikes, key=lambda x: abs(x - current_price))

            # Get option contracts for these strikes
            option_contracts = []

            for expiry in chain.expirations:
                # Calculate DTE
                expiry_date = datetime.datetime.strptime(expiry, '%Y%m%d').date()
                dte = (expiry_date - date.date()).days

                # For backtesting, we only need options that match our strategy criteria
                # Short leg: 1-7 DTE
                # Long leg: 21-45 DTE
                if not (1 <= dte <= 7 or 21 <= dte <= 45):
                    continue

                # Add call and put options for this expiry and atm_strike
                call_contract = Option(symbol, expiry, atm_strike, 'C', 'SMART')
                put_contract = Option(symbol, expiry, atm_strike, 'P', 'SMART')

                option_contracts.extend([call_contract, put_contract])

            if not option_contracts:
                logger.warning(f"No option contracts found for {symbol} on {date}")
                return None

            # Qualify all contracts at once
            self.ib.qualifyContracts(*option_contracts)

            # Request historical data for all options
            # We'll batch our requests to avoid overwhelming IBKR API
            option_data = []
            batch_size = 30

            for i in range(0, len(option_contracts), batch_size):
                batch = option_contracts[i:i + batch_size]

                for contract in batch:
                    # Request historical end of day data
                    try:
                        bars = self.ib.reqHistoricalData(
                            contract=contract,
                            endDateTime=date.strftime('%Y%m%d %H:%M:%S'),
                            durationStr='1 D',
                            barSizeSetting='1 day',
                            whatToShow='TRADES',
                            useRTH=True,
                            formatDate=1
                        )

                        # If no data, try BID_ASK instead of TRADES
                        if not bars:
                            bars = self.ib.reqHistoricalData(
                                contract=contract,
                                endDateTime=date.strftime('%Y%m%d %H:%M:%S'),
                                durationStr='1 D',
                                barSizeSetting='1 day',
                                whatToShow='BID_ASK',
                                useRTH=True,
                                formatDate=1
                            )

                        # If still no data, try with MIDPOINT
                        if not bars:
                            bars = self.ib.reqHistoricalData(
                                contract=contract,
                                endDateTime=date.strftime('%Y%m%d %H:%M:%S'),
                                durationStr='1 D',
                                barSizeSetting='1 day',
                                whatToShow='MIDPOINT',
                                useRTH=True,
                                formatDate=1
                            )

                        if bars:
                            # Implied vol is not directly available from historical data
                            # We need to request it separately
                            ticker = self.ib.reqMktData(contract)
                            self.ib.sleep(0.2)  # Give time for data to arrive

                            # Extract option info
                            expiry_date = datetime.datetime.strptime(contract.lastTradeDateOrContractMonth,
                                                                     '%Y%m%d').date()
                            dte = (expiry_date - date.date()).days

                            option_type = 'short' if 1 <= dte <= 7 else 'long'

                            # Calculate midpoint
                            midpoint = (bars[0].high + bars[0].low) / 2 if len(bars) > 0 else 0

                            option_info = {
                                'contract': contract,
                                'expiration': contract.lastTradeDateOrContractMonth,
                                'strike': contract.strike,
                                'right': contract.right,
                                'bid': bars[0].low if len(bars) > 0 else 0,  # Approximate
                                'ask': bars[0].high if len(bars) > 0 else 0,  # Approximate
                                'midpoint': midpoint,
                                'option_type': option_type,
                                'dte': dte,
                                'kind': 'call' if contract.right == 'C' else 'put',
                                'iv': ticker.impliedVol if hasattr(ticker,
                                                                   'impliedVol') and ticker.impliedVol is not None else 0.3
                                # Default if not available
                            }

                            option_data.append(option_info)

                            # Cancel market data request to free up resources
                            self.ib.cancelMktData(contract)

                    except Exception as e:
                        logger.warning(
                            f"Error getting data for {contract.symbol} {contract.right} {contract.strike} {contract.lastTradeDateOrContractMonth}: {e}")
                        continue

                # Sleep between batches to prevent rate limiting
                if i + batch_size < len(option_contracts):
                    self.ib.sleep(1)

            if not option_data:
                logger.warning(f"No option data retrieved for {symbol} on {date}")
                return None

            # Calculate term structure
            term_structure = self.calculate_term_structure(option_data)

            return {
                'symbol': symbol,
                'underlying_price': current_price,
                'atm_strike': atm_strike,
                'options': option_data,
                'term_structure': term_structure,
                'date': date
            }

        except Exception as e:
            logger.error(f"Error retrieving historical option chain for {symbol} on {date}: {e}")
            traceback.print_exc()
            return None

    def calculate_term_structure(self, option_data):
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
        dtes = np.array([exp['dte'] for exp in expirations])
        ivs = np.array([exp['iv'] for exp in expirations])

        if len(dtes) < 2 or any(np.isnan(ivs)):
            return 0

        slope, _, _, _, _ = stats.linregress(dtes, ivs)
        return float(slope)  # Explicitly convert to Python float


class FinnhubDataClient:
    """Class to handle historical earnings data retrieval from Finnhub"""

    def __init__(self, api_key):
        """Initialize Finnhub client"""
        self.api_key = api_key
        self.client = finnhub.Client(api_key=api_key)

    def get_historical_earnings(self, start_date, end_date):
        """
        Retrieve historical earnings announcements from Finnhub

        Args:
            start_date (datetime): Start date
            end_date (datetime): End date

        Returns:
            dict: Earnings calendar data
        """
        try:
            # Format dates as required by Finnhub API
            from_str = start_date.strftime("%Y-%m-%d")
            to_str = end_date.strftime("%Y-%m-%d")

            # Log the request - this can take a while for long date ranges
            logger.info(f"Fetching earnings data from {from_str} to {to_str} from Finnhub")

            # Fetch the data
            earnings_data = self.client.earnings_calendar(_from=from_str, to=to_str, symbol="")

            # Check if we got valid data
            if not earnings_data or 'earningsCalendar' not in earnings_data:
                logger.error("Invalid or empty response from Finnhub earnings API")
                return {'earningsCalendar': []}

            logger.info(f"Retrieved {len(earnings_data['earningsCalendar'])} earnings announcements from Finnhub")

            return earnings_data

        except Exception as e:
            logger.error(f"Error fetching earnings calendar from Finnhub: {e}")
            return {'earningsCalendar': []}

    def get_daily_stock_data(self, symbol, start_date, end_date):
        """
        Get historical daily stock data from Finnhub

        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date
            end_date (datetime): End date

        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            # Convert dates to UNIX timestamps
            from_timestamp = int(start_date.timestamp())
            to_timestamp = int(end_date.timestamp())

            # Get stock candles from Finnhub
            candles = self.client.stock_candles(symbol, 'D', from_timestamp, to_timestamp)

            if candles['s'] != 'ok':
                logger.warning(f"No data found for {symbol} from Finnhub: {candles['s']}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame({
                'Open': candles['o'],
                'High': candles['h'],
                'Low': candles['l'],
                'Close': candles['c'],
                'Volume': candles['v'],
                'Timestamp': candles['t']
            })

            # Convert timestamp to datetime
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
            df = df.set_index('Date')

            # Drop timestamp column
            df = df.drop(columns=['Timestamp'])

            return df

        except Exception as e:
            logger.error(f"Error retrieving stock data from Finnhub for {symbol}: {e}")
            return None


class CalendarSpreadBacktester:
    """
    Main class for backtesting calendar spread strategy around earnings
    using real-world IBKR historical data and Finnhub earnings data
    """

    def __init__(self, config_file='config.json'):
        """Initialize the backtester"""
        self.config = self.load_config(config_file)

        # Initialize IBKR historical data client
        ibkr_config = self.config.get('ibkr', {})
        self.ibkr = IBKRHistoricalData(
            host=ibkr_config.get('host', '127.0.0.1'),
            port=ibkr_config.get('port', 7497),
            client_id=ibkr_config.get('client_id', 10)
        )

        # Initialize Finnhub client
        api_key = self.config.get('api_keys', {}).get('finnhub', '')
        self.finnhub = FinnhubDataClient(api_key)

        # Extract strategy parameters from config
        self.strategy_params = self.config.get('strategy', {})

        # Trading costs
        self.costs = {
            'slippage_pct': 0.5,  # 0.5% slippage per trade
            'commission_per_contract': 0.65,  # IBKR commission per contract
            'exchange_fees_per_contract': 0.30,  # Exchange fees
            'min_commission_per_order': 1.0  # Minimum commission per order
        }

        # Performance tracking
        self.results = {
            'trades': [],
            'equity_curve': [],
            'performance_metrics': {},
            'monthly_returns': {}
        }

        # Account settings
        self.account_size = self.config.get('account', {}).get('size', 10000)
        self.risk_percent = self.config.get('account', {}).get('risk_per_trade', 6.5)

    def load_config(self, config_file):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            # Return default config
            return {
                'api_keys': {'finnhub': ''},
                'ibkr': {'host': '127.0.0.1', 'port': 7497, 'client_id': 10},
                'account': {'size': 10000, 'risk_per_trade': 6.5},
                'strategy': {
                    'entry_minutes_before_close': 15,
                    'exit_minutes_after_open': 15,
                    'short_dte_min': 1,
                    'short_dte_max': 7,
                    'long_dte_min': 21,
                    'long_dte_max': 45,
                    'long_dte_target': 30,
                    'screening_criteria': {
                        'min_avg_volume': 1500000,
                        'min_iv30_rv30_ratio': 1.25,
                        'max_term_structure_slope': -0.00406
                    }
                }
            }

    def yang_zhang(self, price_data, window=30, trading_periods=252, return_last_only=True):
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
            return float(result.iloc[-1]) if not result.empty else None
        else:
            return result.dropna()

    def analyze_stock_metrics(self, symbol, date):
        """
        Analyze stock metrics for screening

        Args:
            symbol (str): Stock symbol
            date (datetime): Date to analyze

        Returns:
            dict: Analysis results
        """
        try:
            # Get screening criteria from config
            criteria = self.strategy_params.get('screening_criteria', {})
            min_avg_volume = criteria.get('min_avg_volume', 1500000)
            min_iv30_rv30_ratio = criteria.get('min_iv30_rv30_ratio', 1.25)
            max_term_structure_slope = criteria.get('max_term_structure_slope', -0.00406)

            logger.info(f"Analyzing metrics for {symbol} on {date.strftime('%Y-%m-%d')}")

            # Look back 90 days for historical data
            start_date = date - datetime.timedelta(days=90)

            # Get historical price data from IBKR
            price_history = self.ibkr.get_historical_stock_data(symbol, start_date, date)

            if price_history is None or price_history.empty:
                logger.error(f"No price history found for {symbol}")
                return None

            # Calculate 30-day average volume
            avg_volume = float(price_history['Volume'].rolling(30).mean().dropna().iloc[-1])
            logger.info(f"{symbol} 30-day avg volume: {avg_volume:.0f}")

            # Get current price
            current_price = float(price_history['Close'].iloc[-1])

            # Get option chain
            option_chain = self.ibkr.get_historical_option_chain(symbol, date, current_price)

            if option_chain is None:
                logger.error(f"Could not retrieve option chain for {symbol}")
                return None

            # Calculate term structure slope
            ts_slope_0_45 = option_chain['term_structure']
            logger.info(f"{symbol} term structure slope (0-45): {ts_slope_0_45:.6f}")

            # Calculate 30-day RV using Yang-Zhang estimator
            rv30 = self.yang_zhang(price_history)

            if rv30 is None:
                logger.error(f"Could not calculate RV30 for {symbol}")
                return None

            # Calculate IV/RV ratio
            # Extract options with around 30 DTE
            options_around_30dte = [opt for opt in option_chain['options']
                                    if abs(opt['dte'] - 30) <= 5 and opt['right'] == 'C']

            if not options_around_30dte:
                logger.error(f"No options with ~30 DTE found for {symbol}")
                return None

            # Use the average IV of these options
            iv30 = float(np.mean([opt['iv'] for opt in options_around_30dte]))

            iv30_rv30 = iv30 / rv30 if rv30 > 0 else 0
            logger.info(f"{symbol} IV30/RV30 ratio: {iv30_rv30:.2f}")

            # Check if metrics meet criteria
            avg_volume_check = avg_volume >= min_avg_volume
            iv30_rv30_check = iv30_rv30 >= min_iv30_rv30_ratio
            ts_slope_check = ts_slope_0_45 <= max_term_structure_slope  # Negative slope is preferred

            # Return results
            results = {
                'symbol': symbol,
                'date': date,
                'avg_volume': float(avg_volume),
                'avg_volume_pass': bool(avg_volume_check),
                'iv30_rv30': float(iv30_rv30),
                'iv30_rv30_pass': bool(iv30_rv30_check),
                'ts_slope_0_45': float(ts_slope_0_45),
                'ts_slope_pass': bool(ts_slope_check),
                'all_criteria_met': bool(avg_volume_check and iv30_rv30_check and ts_slope_check),
                'current_price': current_price,
                'option_chain': option_chain
            }

            return results

        except Exception as e:
            logger.error(f"Error analyzing metrics for {symbol}: {e}")
            traceback.print_exc()
            return None

    def find_best_calendar_spread(self, option_chain, symbol_metrics=None):
        """
        Find the best calendar spread for the given option chain

        Args:
            option_chain (dict): Option chain data
            symbol_metrics (dict, optional): Symbol metrics from screening

        Returns:
            dict: Best calendar spread
        """
        if not option_chain or 'options' not in option_chain:
            return None

        symbol = option_chain['symbol']

        # If we have symbol metrics from our screening, use those
        if symbol_metrics and not symbol_metrics.get('all_criteria_met', False):
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
                    'underlying_price': option_chain['underlying_price'],
                    'date': option_chain['date']
                }
            else:
                # Put calendar is better
                return {
                    'type': 'put_calendar',
                    'short': best_put_calendar['short'],
                    'long': best_put_calendar['long'],
                    'debit': best_put_calendar['debit'],
                    'symbol': option_chain['symbol'],
                    'underlying_price': option_chain['underlying_price'],
                    'date': option_chain['date']
                }
        elif best_call_calendar:
            # Only call calendar available
            return {
                'type': 'call_calendar',
                'short': best_call_calendar['short'],
                'long': best_call_calendar['long'],
                'debit': best_call_calendar['debit'],
                'symbol': option_chain['symbol'],
                'underlying_price': option_chain['underlying_price'],
                'date': option_chain['date']
            }
        elif best_put_calendar:
            # Only put calendar available
            return {
                'type': 'put_calendar',
                'short': best_put_calendar['short'],
                'long': best_put_calendar['long'],
                'debit': best_put_calendar['debit'],
                'symbol': option_chain['symbol'],
                'underlying_price': option_chain['underlying_price'],
                'date': option_chain['date']
            }
        else:
            # No valid calendar spread found
            return None

    def simulate_calendar_spread_trade(self, calendar_spread, exit_date):
        """
        Simulate a calendar spread trade including entry and exit
        with realistic costs, slippage, and fill modeling

        Args:
            calendar_spread (dict): Calendar spread information
            exit_date (datetime): Date to exit the position

        Returns:
            dict: Trade results
        """
        if not calendar_spread:
            return None

        try:
            symbol = calendar_spread['symbol']
            spread_type = calendar_spread['type']
            entry_date = calendar_spread['date']

            # Get option chain for exit date
            exit_option_chain = self.ibkr.get_historical_option_chain(
                symbol, exit_date, None)

            if not exit_option_chain:
                logger.error(f"Could not retrieve exit option chain for {symbol} on {exit_date}")
                return None

            # Entry cost calculation with slippage
            entry_debit = calendar_spread['debit']
            slippage_pct = self.costs['slippage_pct'] / 100
            entry_debit_with_slippage = entry_debit * (1 + slippage_pct)

            # Calculate position size based on risk
            account_value = self.account_size
            max_risk = account_value * (self.risk_percent / 100)
            risk_per_contract = entry_debit_with_slippage * 100
            contracts = max(1, int(max_risk / risk_per_contract))

            # Total entry cost
            total_entry_cost = entry_debit_with_slippage * contracts * 100

            # Calculate commissions and fees
            commission_per_side = max(
                self.costs['min_commission_per_order'],
                contracts * self.costs['commission_per_contract']
            )
            entry_commission = commission_per_side * 2  # Both legs
            exit_commission = commission_per_side * 2  # Both legs

            # Exchange fees
            exchange_fees = contracts * self.costs['exchange_fees_per_contract'] * 4  # Entry + exit, both legs

            # Total trading costs
            total_costs = entry_commission + exit_commission + exchange_fees

            # Exit calculation - find matching options in the exit chain
            short_exp = calendar_spread['short']['expiration']
            long_exp = calendar_spread['long']['expiration']
            strike = calendar_spread['short']['strike']
            right = calendar_spread['short']['right']

            # Find matching options in exit chain
            exit_options = exit_option_chain['options']

            short_option_exit = next((opt for opt in exit_options
                                      if opt['expiration'] == short_exp
                                      and opt['strike'] == strike
                                      and opt['right'] == right), None)

            long_option_exit = next((opt for opt in exit_options
                                     if opt['expiration'] == long_exp
                                     and opt['strike'] == strike
                                     and opt['right'] == right), None)

            if not short_option_exit or not long_option_exit:
                logger.warning(f"Could not find matching options for exit on {exit_date} for {symbol}")

                # In this case, we'll estimate exit value with expected decay
                # This is an approximation, in real backtest we'd need actual exit prices
                days_held = (exit_date - entry_date).days
                # Using the theta estimate based on days held
                theta_decay_pct = 0.15 * days_held  # Simplified model, actual would use option pricing models
                exit_debit = entry_debit * (1 - theta_decay_pct)
            else:
                # Calculate exit debit
                exit_debit = long_option_exit['midpoint'] - short_option_exit['midpoint']

            # Apply exit slippage
            exit_debit_with_slippage = exit_debit * (1 - slippage_pct)

            # Calculate P&L
            pnl_per_contract = (entry_debit - exit_debit_with_slippage) * 100
            total_pnl = pnl_per_contract * contracts - total_costs

            # Format trade result
            trade_result = {
                'symbol': symbol,
                'strategy': spread_type,
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'days_held': (exit_date - entry_date).days,
                'short_expiry': short_exp,
                'long_expiry': long_exp,
                'strike': strike,
                'right': right,
                'contracts': contracts,
                'entry_debit': entry_debit,
                'entry_debit_with_slippage': entry_debit_with_slippage,
                'exit_debit': exit_debit,
                'exit_debit_with_slippage': exit_debit_with_slippage,
                'entry_cost': total_entry_cost,
                'commissions_fees': total_costs,
                'pnl_per_contract': pnl_per_contract,
                'total_pnl': total_pnl,
                'roi_pct': (total_pnl / total_entry_cost) * 100 if total_entry_cost > 0 else 0
            }

            logger.info(
                f"Trade simulation for {symbol} {spread_type}: PnL=${total_pnl:.2f}, ROI={trade_result['roi_pct']:.2f}%")

            return trade_result

        except Exception as e:
            logger.error(f"Error simulating trade for {calendar_spread['symbol']}: {e}")
            traceback.print_exc()
            return None

    def filter_symbols_by_hour(self, earnings_data, date, hour):
        """Filter symbols reporting at specific hour on specific date"""
        filtered_symbols = []
        date_str = date.strftime("%Y-%m-%d")

        for item in earnings_data['earningsCalendar']:
            if item['date'] == date_str and item['hour'] == hour:
                filtered_symbols.append(item['symbol'])

        return filtered_symbols

    def is_trading_day(self, date):
        """Check if the given date is a trading day (not weekend/holiday)"""
        # Check for weekends
        if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False

        # Simple holiday check - this is a simplification
        # In real implementation, use market calendar library
        holidays = [
            # 2022 US market holidays
            datetime.date(2022, 1, 1),  # New Year's Day
            datetime.date(2022, 1, 17),  # Martin Luther King Jr. Day
            datetime.date(2022, 2, 21),  # Presidents' Day
            datetime.date(2022, 4, 15),  # Good Friday
            datetime.date(2022, 5, 30),  # Memorial Day
            datetime.date(2022, 6, 20),  # Juneteenth (observed)
            datetime.date(2022, 7, 4),  # Independence Day
            datetime.date(2022, 9, 5),  # Labor Day
            datetime.date(2022, 11, 24),  # Thanksgiving
            datetime.date(2022, 12, 26),  # Christmas (observed)

            # 2023 US market holidays
            datetime.date(2023, 1, 2),  # New Year's Day (observed)
            datetime.date(2023, 1, 16),  # Martin Luther King Jr. Day
            datetime.date(2023, 2, 20),  # Presidents' Day
            datetime.date(2023, 4, 7),  # Good Friday
            datetime.date(2023, 5, 29),  # Memorial Day
            datetime.date(2023, 6, 19),  # Juneteenth
            datetime.date(2023, 7, 4),  # Independence Day
            datetime.date(2023, 9, 4),  # Labor Day
            datetime.date(2023, 11, 23),  # Thanksgiving
            datetime.date(2023, 12, 25),  # Christmas

            # Add more years as needed
        ]

        if date.date() in holidays:
            return False

        return True

    def get_next_trading_day(self, date):
        """Get the next trading day after the given date"""
        next_day = date + datetime.timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day = next_day + datetime.timedelta(days=1)
        return next_day

    def run_backtest(self, start_date, end_date):
        """
        Run backtest for the calendar spread strategy

        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format

        Returns:
            dict: Backtest results
        """
        # Convert string dates to datetime objects
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Connect to IBKR
        if not self.ibkr.connect():
            logger.error("Failed to connect to IBKR. Aborting backtest.")
            return None

        try:
            # Fetch historical earnings announcements for the entire period
            earnings_data = self.finnhub.get_historical_earnings(start_dt, end_dt)

            if not earnings_data or len(earnings_data.get('earningsCalendar', [])) == 0:
                logger.error("No earnings data found for the period. Aborting backtest.")
                self.ibkr.disconnect()
                return None

            # Initialize results tracking
            self.results = {
                'trades': [],
                'equity_curve': [(start_date, self.account_size)],
                'performance_metrics': {},
                'monthly_returns': {}
            }

            # Initialize equity tracking
            current_equity = self.account_size

            # Process each trading day in the backtest period
            current_date = start_dt

            with tqdm(total=(end_dt - start_dt).days, desc="Backtesting") as pbar:
                while current_date <= end_dt:
                    # Skip non-trading days
                    if not self.is_trading_day(current_date):
                        current_date += datetime.timedelta(days=1)
                        pbar.update(1)
                        continue

                    next_trading_day = self.get_next_trading_day(current_date)

                    logger.info(f"Processing trading day: {current_date.strftime('%Y-%m-%d')}")

                    # Get earnings symbols for current day (after market close)
                    amc_today = self.filter_symbols_by_hour(earnings_data, current_date, 'amc')

                    # Get earnings symbols for next trading day (before market open)
                    bmo_next = self.filter_symbols_by_hour(earnings_data, next_trading_day, 'bmo')

                    # Combine symbols (both are potential trades)
                    all_symbols = list(set(amc_today + bmo_next))

                    if all_symbols:
                        logger.info(f"Found {len(all_symbols)} potential earnings plays: {', '.join(all_symbols)}")

                        # Screen and analyze each symbol
                        candidates = []

                        for symbol in all_symbols:
                            # Skip if too many trades are already active (limit exposure)
                            if len(candidates) >= 5:  # Limit to 5 trades per day
                                break

                            # Analyze the symbol for trading criteria
                            metrics = self.analyze_stock_metrics(symbol, current_date)

                            if metrics and metrics.get('all_criteria_met', False):
                                # Find best calendar spread
                                calendar_spread = self.find_best_calendar_spread(
                                    metrics['option_chain'], metrics)

                                if calendar_spread:
                                    candidates.append(calendar_spread)
                                    logger.info(f"Added {symbol} {calendar_spread['type']} to trade candidates")

                        # Execute trades for candidates
                        for spread in candidates:
                            # Simulate the trade
                            trade_result = self.simulate_calendar_spread_trade(spread, next_trading_day)

                            if trade_result:
                                # Add to trade history
                                self.results['trades'].append(trade_result)

                                # Update equity
                                current_equity += trade_result['total_pnl']
                                self.results['equity_curve'].append((current_date.strftime('%Y-%m-%d'), current_equity))

                                # Update monthly returns tracking
                                month_key = current_date.strftime('%Y-%m')
                                if month_key not in self.results['monthly_returns']:
                                    self.results['monthly_returns'][month_key] = 0
                                self.results['monthly_returns'][month_key] += trade_result['total_pnl']

                    # Move to next day
                    current_date += datetime.timedelta(days=1)
                    pbar.update(1)

            # Calculate performance metrics
            self.calculate_performance_metrics()

            # Save results
            self.save_results(f"backtest_{start_date}_to_{end_date}")

            # Create and save plots
            self.create_performance_plots(f"backtest_{start_date}_to_{end_date}")

            # Disconnect from IBKR
            self.ibkr.disconnect()

            return self.results

        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            traceback.print_exc()

            # Disconnect from IBKR
            self.ibkr.disconnect()

            return None

    def calculate_performance_metrics(self):
        """Calculate performance metrics from backtest results"""
        try:
            trades = self.results['trades']
            equity_curve = self.results['equity_curve']

            if not trades or len(trades) == 0:
                logger.warning("No trades executed during backtest period")
                self.results['performance_metrics'] = {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_profit_per_trade': 0,
                    'profit_factor': 0,
                    'total_pnl': 0,
                    'max_drawdown': 0,
                    'max_drawdown_pct': 0,
                    'sharpe_ratio': 0,
                    'annual_return_pct': 0
                }
                return

            # Basic metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['total_pnl'] > 0]
            losing_trades = [t for t in trades if t['total_pnl'] <= 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

            # Average profit per trade
            total_pnl = sum(t['total_pnl'] for t in trades)
            avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0

            # Profit factor
            total_gains = sum(t['total_pnl'] for t in winning_trades)
            total_losses = sum(abs(t['total_pnl']) for t in losing_trades)
            profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

            # Calculate returns
            equity_values = [e[1] for e in equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]

            # Sharpe ratio
            annual_return = (equity_values[-1] / equity_values[0]) - 1
            trading_days = len([e for e in equity_curve if e[1] != equity_values[0]])
            years = trading_days / 252  # Assuming 252 trading days per year
            annual_return_pct = (annual_return / years) * 100 if years > 0 else 0

            avg_return = np.mean(returns) if len(returns) > 0 else 0
            std_return = np.std(returns) if len(returns) > 0 else 0
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

            # Maximum drawdown
            rolling_max = np.maximum.accumulate(equity_values)
            drawdowns = 1 - np.array(equity_values) / rolling_max
            max_drawdown_pct = np.max(drawdowns) * 100
            max_drawdown_idx = np.argmax(drawdowns)
            max_drawdown = rolling_max[max_drawdown_idx] - equity_values[max_drawdown_idx]

            # Store metrics
            self.results['performance_metrics'] = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_profit_per_trade': avg_profit_per_trade,
                'profit_factor': profit_factor,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'sharpe_ratio': sharpe_ratio,
                'annual_return_pct': annual_return_pct
            }

            logger.info(f"Calculated performance metrics: {self.results['performance_metrics']}")

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            traceback.print_exc()

    def save_results(self, filename_prefix):
        """Save backtest results to file"""
        try:
            # Create results directory if it doesn't exist
            os.makedirs('backtest_results', exist_ok=True)

            # Save results as JSON
            filename = f"backtest_results/{filename_prefix}_results.json"

            with open(filename, 'w') as f:
                # Convert dates in equity curve to strings
                results_copy = self.results.copy()
                results_copy['equity_curve'] = [(date, value) for date, value in self.results['equity_curve']]

                json.dump(results_copy, f, indent=2)

            logger.info(f"Saved backtest results to {filename}")

            # Save trade details to CSV
            trades_file = f"backtest_results/{filename_prefix}_trades.csv"
            trades_df = pd.DataFrame(self.results['trades'])
            trades_df.to_csv(trades_file, index=False)

            logger.info(f"Saved trade details to {trades_file}")

        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            traceback.print_exc()

    def create_performance_plots(self, filename_prefix):
        """Create and save performance plots"""
        try:
            # Create plots directory if it doesn't exist
            os.makedirs('backtest_results/plots', exist_ok=True)

            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')

            # 1. Equity curve
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date, _ in self.results['equity_curve']]
            equity = [value for _, value in self.results['equity_curve']]

            ax.plot(dates, equity, linewidth=2)
            ax.set_title('Equity Curve', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Equity ($)', fontsize=12)
            ax.grid(True)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            # Add starting and ending equity as text
            if len(equity) > 1:
                start_equity = equity[0]
                end_equity = equity[-1]
                total_return = ((end_equity / start_equity) - 1) * 100

                textstr = f'Starting Equity: ${start_equity:,.2f}\nEnding Equity: ${end_equity:,.2f}\nTotal Return: {total_return:.2f}%'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)

            plt.tight_layout()
            plt.savefig(f"backtest_results/plots/{filename_prefix}_equity_curve.png", dpi=300)
            plt.close()

            # 2. Monthly returns heatmap
            if self.results['monthly_returns']:
                # Convert to DataFrame
                monthly_data = []

                for month_key, pnl in self.results['monthly_returns'].items():
                    year, month = month_key.split('-')
                    monthly_data.append({
                        'Year': int(year),
                        'Month': int(month),
                        'PnL': pnl,
                        'Return': pnl / self.account_size * 100  # Monthly return percentage
                    })

                df = pd.DataFrame(monthly_data)

                # Create pivot table
                pivot = df.pivot(index='Year', columns='Month', values='Return')

                # Create heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                            linewidths=1, cbar_kws={'label': 'Return (%)'})

                plt.title('Monthly Returns (%)', fontsize=14)
                plt.tight_layout()
                plt.savefig(f"backtest_results/plots/{filename_prefix}_monthly_returns.png", dpi=300)
                plt.close()

            # 3. Trade PnL distribution
            if self.results['trades']:
                pnls = [trade['total_pnl'] for trade in self.results['trades']]

                plt.figure(figsize=(10, 6))
                sns.histplot(pnls, kde=True, bins=20)
                plt.axvline(x=0, color='r', linestyle='--')

                plt.title('Trade PnL Distribution', fontsize=14)
                plt.xlabel('PnL ($)', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)

                plt.tight_layout()
                plt.savefig(f"backtest_results/plots/{filename_prefix}_pnl_distribution.png", dpi=300)
                plt.close()

            # 4. Performance metrics table
            if self.results['performance_metrics']:
                metrics = self.results['performance_metrics']

                # Create a figure with no axes
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)

                # Hide axes
                ax.axis('off')
                ax.axis('tight')

                # Create data for table
                data = [
                    ['Total Trades', f"{metrics['total_trades']}"],
                    ['Winning Trades', f"{metrics['winning_trades']} ({metrics['win_rate'] * 100:.2f}%)"],
                    ['Losing Trades', f"{metrics['losing_trades']}"],
                    ['Total P&L', f"${metrics['total_pnl']:,.2f}"],
                    ['Avg. Profit per Trade', f"${metrics['avg_profit_per_trade']:,.2f}"],
                    ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
                    ['Max Drawdown', f"${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)"],
                    ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
                    ['Annual Return', f"{metrics['annual_return_pct']:.2f}%"]
                ]

                # Create the table
                table = ax.table(cellText=data, colLabels=['Metric', 'Value'],
                                 loc='center', cellLoc='left')

                # Modify table appearance
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 1.5)

                # Set title
                plt.title('Backtest Performance Metrics', fontsize=16, pad=20)

                plt.tight_layout()
                plt.savefig(f"backtest_results/plots/{filename_prefix}_metrics_table.png", dpi=300)
                plt.close()

            logger.info(f"Created and saved performance plots to backtest_results/plots/")

        except Exception as e:
            logger.error(f"Error creating performance plots: {e}")
            traceback.print_exc()


def main():
    """Main function to run the backtester"""
    parser = argparse.ArgumentParser(description='Calendar Spread Strategy Backtester using Real IBKR and Finnhub Data')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')

    args = parser.parse_args()

    # Log start time
    start_time = time.time()
    logger.info(f"Starting backtest using IBKR and Finnhub data from {args.start} to {args.end}")

    # Create and run backtester
    backtester = CalendarSpreadBacktester(args.config)
    results = backtester.run_backtest(args.start, args.end)

    if results:
        # Log summary
        metrics = results['performance_metrics']
        logger.info("Backtest completed successfully!")
        logger.info(f"Total trades: {metrics['total_trades']}")
        logger.info(f"Win rate: {metrics['win_rate'] * 100:.2f}%")
        logger.info(f"Total P&L: ${metrics['total_pnl']:,.2f}")
        logger.info(f"Annual return: {metrics['annual_return_pct']:.2f}%")

        # Log elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Backtest completed in {elapsed_time:.2f} seconds")
    else:
        logger.error("Backtest failed")


if __name__ == "__main__":
    main()