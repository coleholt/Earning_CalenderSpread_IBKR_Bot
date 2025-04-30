"""
IBKR Data Collector for Earnings Calendar Bot

This module replaces yfinance API calls with IBKR's API via ib_insync to avoid rate limits.
It collects stock price history, volatility data, and options chains directly from IBKR.
"""

import logging
import datetime
import time
import pandas as pd
import numpy as np
from ib_insync import *
from scipy import stats
from scipy.interpolate import interp1d
import os
import json

logger = logging.getLogger(__name__)

# Create cache directory
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class IBKRDataCollector:
    """Collects stock and options data from IBKR to replace yfinance"""

    def __init__(self, ib=None):
        """
        Initialize with an existing IB connection or create a new one.

        Args:
            ib: An existing ib_insync.IB instance, or None to create a new one
        """
        self.ib = ib
        self._connection_owned = (ib is None)

    def ensure_connection(self):
        """Ensure we have an active IBKR connection"""
        if self.ib is None:
            # Create a new connection using parameters from config
            from earnings_calendar_bot import load_config
            config = load_config()
            ibkr_config = config.get('ibkr', {})
            host = ibkr_config.get('host', '127.0.0.1')
            port = ibkr_config.get('port', 7497)
            client_id = ibkr_config.get('client_id', 1)

            logger.info(f"Connecting to IBKR at {host}:{port} with client ID {client_id}")
            self.ib = IB()
            self.ib.connect(host, port, clientId=client_id)
            self._connection_owned = True
            logger.info("Connected to IBKR")

        if not self.ib.isConnected():
            logger.warning("IBKR connection lost, reconnecting...")
            self.ib.connect()

    def disconnect(self):
        """Disconnect from IBKR if we own the connection"""
        if self._connection_owned and self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")
            self.ib = None

    def get_price_history(self, symbol, duration='3 M', bar_size='1 day'):
        """
        Get price history for a stock (replacement for yfinance history)

        Args:
            symbol: Stock symbol
            duration: History duration (e.g., '3 M' for 3 months)
            bar_size: Bar size (e.g., '1 day' for daily bars)

        Returns:
            pandas.DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        # Check cache first
        cache_key = f"{symbol}_{duration.replace(' ', '')}_{bar_size.replace(' ', '')}"
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}_price_history.csv")

        # Use cache if it exists and is recent (less than 1 day old)
        if os.path.exists(cache_file):
            file_modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.datetime.now() - file_modified_time).total_seconds() < 86400:  # 24 hours
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    logger.info(f"Loaded price history for {symbol} from cache")
                    return df
                except Exception as e:
                    logger.warning(f"Error loading cached price history: {e}")

        # If not in cache or cache is old, get from IBKR
        self.ensure_connection()

        try:
            # Create a stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',  # Empty for latest data
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )

            if not bars:
                logger.error(f"No price history found for {symbol}")
                return None

            # Convert to DataFrame
            df = util.df(bars)

            # Rename columns to match yfinance format
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Set date as index
            df = df.set_index('Date')

            # Cache the result
            try:
                df.to_csv(cache_file)
            except Exception as e:
                logger.warning(f"Error writing price history to cache: {e}")

            return df

        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return None

    def get_option_chain(self, symbol):
        """
        Get current option chain for a stock (replacement for yfinance option_chain)

        Args:
            symbol: Stock symbol

        Returns:
            dict with keys: calls, puts, underlying_price
        """
        self.ensure_connection()

        try:
            # Create a stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)

            # Get current price
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(0.2)  # Give a moment for market data to arrive

            price = ticker.marketPrice()
            if not price or price <= 0:
                price = ticker.close
            if not price or price <= 0:
                price = ticker.last

            if not price or price <= 0:
                logger.error(f"Could not determine price for {symbol}")
                self.ib.cancelMktData(contract)
                return None

            self.ib.cancelMktData(contract)

            # Request option chain
            chains = self.ib.reqSecDefOptParams(contract.symbol, '', contract.secType, contract.conId)

            if not chains:
                logger.error(f"No option chains found for {symbol}")
                return None

            # Filter for options on the SMART exchange
            chain = next((c for c in chains if c.exchange == 'SMART'), None)
            if not chain:
                logger.error(f"No SMART exchange options found for {symbol}")
                return None

            # Build a structure similar to yfinance's option_chain
            result = {
                'underlying_price': price,
                'expirations': sorted(chain.expirations),
                'strikes': sorted(chain.strikes)
            }

            # Find the strike closest to current price
            atm_strike = min(chain.strikes, key=lambda x: abs(x - price))
            result['atm_strike'] = atm_strike

            return result

        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return None

    def get_option_chain_for_expiration(self, symbol, expiration):
        """
        Get option chain for a specific expiration date

        Args:
            symbol: Stock symbol
            expiration: Expiration date string (format used by IBKR, e.g., '20250516')

        Returns:
            dict with keys: calls, puts
            Each is a DataFrame with option data
        """
        self.ensure_connection()

        try:
            # Get basic option chain first
            chain_data = self.get_option_chain(symbol)
            if not chain_data:
                return None

            underlying_price = chain_data['underlying_price']

            # Request option contracts for this expiration
            call_contracts = []
            put_contracts = []

            # For efficiency, focus on strikes near the money
            # Calculate a range of strikes to request (e.g., ±20% of current price)
            min_strike = underlying_price * 0.8
            max_strike = underlying_price * 1.2
            relevant_strikes = [s for s in chain_data['strikes'] if min_strike <= s <= max_strike]

            # If we don't have enough strikes in range, use all available strikes
            if len(relevant_strikes) < 10:
                relevant_strikes = chain_data['strikes']

            # Limit to a reasonable number of strikes to avoid too many requests
            if len(relevant_strikes) > 20:
                # Sort by distance from underlying price
                relevant_strikes = sorted(relevant_strikes, key=lambda s: abs(s - underlying_price))
                relevant_strikes = relevant_strikes[:20]

            # Create option contracts
            for strike in relevant_strikes:
                call_contracts.append(Option(symbol, expiration, strike, 'C', 'SMART'))
                put_contracts.append(Option(symbol, expiration, strike, 'P', 'SMART'))

            # Qualify all contracts at once
            all_contracts = call_contracts + put_contracts
            self.ib.qualifyContracts(*all_contracts)

            # Request market data for all options
            tickers = self.ib.reqTickers(*all_contracts)
            self.ib.sleep(0.5)  # Give a moment for market data to arrive

            # Create DataFrames for calls and puts
            calls_data = []
            puts_data = []

            for ticker, contract in zip(tickers, all_contracts):
                # Calculate option data similar to yfinance format
                bid = ticker.bid if ticker.bid > 0 else None
                ask = ticker.ask if ticker.ask > 0 else None
                last = ticker.last if ticker.last > 0 else None

                # Calculate implied volatility
                iv = ticker.impliedVol if ticker.impliedVol > 0 else None

                # Determine a reasonable midpoint price
                if bid and ask:
                    midpoint = (bid + ask) / 2
                elif last:
                    midpoint = last
                else:
                    midpoint = None

                option_data = {
                    'contractSymbol': contract.localSymbol,
                    'strike': contract.strike,
                    'bid': bid,
                    'ask': ask,
                    'last': last,
                    'impliedVolatility': iv,
                    'midpoint': midpoint,
                    'expiration': contract.lastTradeDateOrContractMonth
                }

                if contract.right == 'C':
                    calls_data.append(option_data)
                else:
                    puts_data.append(option_data)

            # Convert to DataFrame
            calls_df = pd.DataFrame(calls_data) if calls_data else pd.DataFrame()
            puts_df = pd.DataFrame(puts_data) if puts_data else pd.DataFrame()

            return {
                'calls': calls_df,
                'puts': puts_df,
                'underlying_price': underlying_price
            }

        except Exception as e:
            logger.error(f"Error getting option chain for {symbol} ({expiration}): {e}")
            return None

    def get_all_options_data(self, symbol):
        """
        Get comprehensive options data for a symbol across multiple expirations

        Args:
            symbol: Stock symbol

        Returns:
            dict with full options data, including implied volatility term structure
        """
        self.ensure_connection()

        try:
            # First get the basic option chain info
            chain_data = self.get_option_chain(symbol)
            if not chain_data:
                return None

            # Filter to a reasonable number of expirations
            expirations = chain_data['expirations']
            if len(expirations) > 10:
                # Keep some short-term, mid-term and long-term expirations
                sorted_exp = sorted(expirations)

                # Take first 5 (short-term), middle 3, and last 2 (long-term)
                short_term = sorted_exp[:5]
                mid_term = sorted_exp[len(sorted_exp) // 2 - 1:len(sorted_exp) // 2 + 2]
                long_term = sorted_exp[-2:]

                expirations = list(set(short_term + mid_term + long_term))

            # Get ATM IV for each expiration to build term structure
            term_structure = {}
            current_price = chain_data['underlying_price']

            for expiration in expirations:
                exp_chain = self.get_option_chain_for_expiration(symbol, expiration)

                if not exp_chain or exp_chain['calls'].empty or exp_chain['puts'].empty:
                    continue

                # Find ATM options
                calls = exp_chain['calls']
                puts = exp_chain['puts']

                if 'strike' not in calls.columns or 'impliedVolatility' not in calls.columns:
                    continue

                # Find closest strikes to current price
                call_diffs = abs(calls['strike'] - current_price)
                call_idx = call_diffs.idxmin() if not call_diffs.empty else None

                put_diffs = abs(puts['strike'] - current_price)
                put_idx = put_diffs.idxmin() if not put_diffs.empty else None

                # Get IVs
                if call_idx is not None and put_idx is not None:
                    call_iv = calls.loc[call_idx, 'impliedVolatility']
                    put_iv = puts.loc[put_idx, 'impliedVolatility']

                    # Average the call and put IVs
                    if call_iv and put_iv:
                        atm_iv = (call_iv + put_iv) / 2.0
                        term_structure[expiration] = atm_iv
                        logger.info(f"Expiration {expiration}: ATM IV = {atm_iv:.4f}")

            # Complete the result with term structure
            chain_data['term_structure'] = term_structure
            return chain_data

        except Exception as e:
            logger.error(f"Error getting all options data for {symbol}: {e}")
            return None

    def get_stock_metrics(self, symbol):
        """
        Calculate stock metrics needed for earnings screening

        This is a direct replacement for the analyze_stock_metrics function
        in the earnings bot, but using IBKR instead of yfinance

        Args:
            symbol: Stock symbol

        Returns:
            dict with metrics similar to the analyze_stock_metrics function
        """
        try:
            # Load screening criteria from config
            from earnings_calendar_bot import load_config
            config = load_config()
            criteria = config['strategy']['screening_criteria']
            min_avg_volume = criteria.get('min_avg_volume', 1500000)
            min_iv30_rv30_ratio = criteria.get('min_iv30_rv30_ratio', 1.25)
            max_term_structure_slope = criteria.get('max_term_structure_slope', -0.00406)

            logger.info(f"Analyzing metrics for {symbol}")

            # Get stock price history
            price_history = self.get_price_history(symbol, duration='3 M', bar_size='1 day')
            if price_history is None or price_history.empty:
                logger.error(f"No price history found for {symbol}")
                return None

            # Calculate 30-day average volume
            avg_volume = float(price_history['Volume'].rolling(30).mean().dropna().iloc[-1])
            logger.info(f"{symbol} 30-day avg volume: {avg_volume:.0f}")

            # Get current price
            current_price = float(price_history['Close'].iloc[-1])

            # Get options data for term structure
            options_data = self.get_all_options_data(symbol)
            if not options_data or not options_data.get('term_structure'):
                logger.error(f"No options data found for {symbol}")
                return None

            # Calculate days to expiry for each date
            today = datetime.datetime.today().date()
            dtes = []
            ivs = []

            for exp_date, iv in options_data['term_structure'].items():
                # Convert expiration from IBKR format (YYYYMMDD) to date object
                exp_date_obj = datetime.datetime.strptime(exp_date, "%Y%m%d").date()
                days_to_expiry = (exp_date_obj - today).days

                # Only use valid data points
                if days_to_expiry > 0 and iv > 0:
                    dtes.append(days_to_expiry)
                    ivs.append(iv)

            # Build term structure model
            if len(dtes) < 2:
                logger.error(f"Not enough IV data points for term structure for {symbol}")
                return None

            # Import the term structure builder function from the earnings bot
            from earnings_calendar_bot import build_term_structure
            term_spline = build_term_structure(dtes, ivs)

            # Calculate term structure slope (0-45 days)
            min_dte = min(dtes)
            ts_slope_0_45 = (term_spline(45) - term_spline(min_dte)) / (45 - min_dte)
            logger.info(f"{symbol} term structure slope (0-45): {ts_slope_0_45:.6f}")

            # Calculate 30-day RV using Yang-Zhang estimator
            from earnings_calendar_bot import yang_zhang
            rv30 = float(yang_zhang(price_history))
            logger.info(f"{symbol} 30-day RV: {rv30:.4f}")

            # Calculate IV/RV ratio
            iv30 = float(term_spline(30))
            logger.info(f"{symbol} 30-day IV: {iv30:.4f}")
            iv30_rv30 = iv30 / rv30 if rv30 > 0 else 0
            logger.info(f"{symbol} IV30/RV30 ratio: {iv30_rv30:.2f}")

            # Check if metrics meet criteria
            avg_volume_check = avg_volume >= min_avg_volume
            iv30_rv30_check = iv30_rv30 >= min_iv30_rv30_ratio
            ts_slope_check = ts_slope_0_45 <= max_term_structure_slope

            logger.info(
                f"Volume check ({avg_volume:.0f} >= {min_avg_volume}): {'PASS' if avg_volume_check else 'FAIL'}")
            logger.info(
                f"IV/RV check ({iv30_rv30:.2f} >= {min_iv30_rv30_ratio}): {'PASS' if iv30_rv30_check else 'FAIL'}")
            logger.info(
                f"Term structure check ({ts_slope_0_45:.6f} <= {max_term_structure_slope}): {'PASS' if ts_slope_check else 'FAIL'}")

            all_criteria_met = avg_volume_check and iv30_rv30_check and ts_slope_check
            logger.info(f"Overall result: {'PASS' if all_criteria_met else 'FAIL'}")

            # Return results
            results = {
                'symbol': symbol,
                'avg_volume': float(avg_volume),
                'avg_volume_pass': bool(avg_volume_check),
                'iv30_rv30': float(iv30_rv30),
                'iv30_rv30_pass': bool(iv30_rv30_check),
                'ts_slope_0_45': float(ts_slope_0_45),
                'ts_slope_pass': bool(ts_slope_check),
                'all_criteria_met': bool(all_criteria_met)
            }

            return results

        except Exception as e:
            logger.error(f"Error analyzing metrics for {symbol}: {e}")
            return None

    def __del__(self):
        """Clean up by disconnecting from IBKR"""
        self.disconnect()


def get_batch_processor(ib=None, batch_size=5):
    """
    Create a batch processor to handle multiple symbols efficiently.

    Args:
        ib: An existing ib_insync.IB instance, or None to create a new one
        batch_size: Number of symbols to process in each batch

    Returns:
        function that processes a list of symbols in batches
    """
    collector = IBKRDataCollector(ib)

    def process_symbols_batch(symbols):
        """
        Process a list of symbols in batches to avoid overwhelming the API

        Args:
            symbols: List of stock symbols to analyze

        Returns:
            dict mapping symbols to their analysis results
        """
        results = {}

        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(symbols) + batch_size - 1) // batch_size}: {', '.join(batch)}")

            for symbol in batch:
                try:
                    metrics = collector.get_stock_metrics(symbol)
                    if metrics:
                        results[symbol] = metrics
                        status = "PASSED" if metrics.get('all_criteria_met', False) else "FAILED"
                        logger.info(f"Analysis complete for {symbol}: {status}")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Small pause between batches to avoid overwhelming TWS/IB Gateway
            if i + batch_size < len(symbols):
                time.sleep(2)

        return results

    return process_symbols_batch


# Example usage for replacing analyze_stock_metrics in earnings_calendar_bot.py
def analyze_stock_metrics_with_ibkr(symbol, ib=None):
    """
    Drop-in replacement for analyze_stock_metrics that uses IBKR instead of yfinance

    Args:
        symbol: Stock symbol to analyze
        ib: Optional existing IB connection

    Returns:
        dict with analysis results
    """
    collector = IBKRDataCollector(ib)
    try:
        return collector.get_stock_metrics(symbol)
    finally:
        if ib is None:  # Only disconnect if we created the connection
            collector.disconnect()