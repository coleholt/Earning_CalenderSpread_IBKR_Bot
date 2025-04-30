#!/usr/bin/env python3
"""
Test script for the IBKR-based stock screening implementation.
This script tests the replacement for yfinance API calls.
"""

import logging
import json
import sys
import time
from ib_insync import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our data collector
try:
    from ibkr_data_collector import IBKRDataCollector
except ImportError:
    logger.error("ibkr_data_collector.py not found. Make sure it's in the current directory.")
    sys.exit(1)


def main():
    """Main test function"""
    logger.info("Testing IBKR-based screening implementation")

    # Connect to IBKR
    ib = IB()
    try:
        logger.info("Connecting to IBKR...")
        ib.connect('127.0.0.1', 7497, clientId=10)  # Use different client ID to avoid conflicts

        if not ib.isConnected():
            logger.error("Failed to connect to IBKR")
            return False

        logger.info("Successfully connected to IBKR")

        # Create data collector
        collector = IBKRDataCollector(ib)

        # Test symbols - include some that should pass the screening criteria
        test_symbols = ["KHC", "AAPL", "MSFT", "AMZN"]

        results = {}

        # Test getting price history
        logger.info("\n=== Testing Price History Function ===")
        for symbol in test_symbols[:2]:  # Just test a couple
            logger.info(f"Getting price history for {symbol}")
            price_history = collector.get_price_history(symbol)

            if price_history is not None and not price_history.empty:
                logger.info(f"✅ Successfully got price history for {symbol}")
                logger.info(f"   Last date: {price_history.index[-1]}")
                logger.info(f"   Last price: ${price_history['Close'].iloc[-1]:.2f}")
                logger.info(f"   Data points: {len(price_history)}")
            else:
                logger.error(f"❌ Failed to get price history for {symbol}")

        # Test getting option chain
        logger.info("\n=== Testing Option Chain Function ===")
        for symbol in test_symbols[:2]:  # Just test a couple
            logger.info(f"Getting option chain for {symbol}")
            chain_data = collector.get_option_chain(symbol)

            if chain_data is not None:
                logger.info(f"✅ Successfully got option chain for {symbol}")
                logger.info(f"   Underlying price: ${chain_data['underlying_price']:.2f}")
                logger.info(f"   Number of expirations: {len(chain_data['expirations'])}")
                logger.info(f"   First few expirations: {chain_data['expirations'][:3]}")
            else:
                logger.error(f"❌ Failed to get option chain for {symbol}")

        # Test options data for specific expiration
        if 'expirations' in chain_data:
            logger.info("\n=== Testing Options Data for Expiration ===")
            first_expiration = chain_data['expirations'][0]

            logger.info(f"Getting options data for {symbol}, expiration {first_expiration}")
            exp_data = collector.get_option_chain_for_expiration(symbol, first_expiration)

            if exp_data is not None:
                logger.info(f"✅ Successfully got options data for {symbol} ({first_expiration})")
                logger.info(f"   Number of calls: {len(exp_data['calls'])}")
                logger.info(f"   Number of puts: {len(exp_data['puts'])}")

                if not exp_data['calls'].empty:
                    logger.info(f"   Call option columns: {list(exp_data['calls'].columns)}")
            else:
                logger.error(f"❌ Failed to get options data for {symbol} ({first_expiration})")

        # Test getting all options data
        logger.info("\n=== Testing Full Options Data Function ===")
        for symbol in test_symbols[:1]:  # Just test one
            logger.info(f"Getting all options data for {symbol}")
            options_data = collector.get_all_options_data(symbol)

            if options_data is not None and 'term_structure' in options_data:
                logger.info(f"✅ Successfully got all options data for {symbol}")
                logger.info(f"   Term structure points: {len(options_data['term_structure'])}")

                # Print first few expirations and IVs
                for i, (exp, iv) in enumerate(list(options_data['term_structure'].items())[:3]):
                    logger.info(f"   Expiration {exp}: IV = {iv:.4f}")
            else:
                logger.error(f"❌ Failed to get all options data for {symbol}")

        # Test full stock metrics analysis
        logger.info("\n=== Testing Full Stock Metrics Analysis ===")
        for symbol in test_symbols:
            logger.info(f"Analyzing metrics for {symbol}")
            metrics = collector.get_stock_metrics(symbol)

            if metrics is not None:
                results[symbol] = metrics
                status = "PASSED" if metrics.get('all_criteria_met', False) else "FAILED"
                logger.info(f"✅ Analysis complete for {symbol}: {status}")
                logger.info(
                    f"   Volume: {metrics['avg_volume']:.0f} - {'PASS' if metrics['avg_volume_pass'] else 'FAIL'}")
                logger.info(f"   IV/RV: {metrics['iv30_rv30']:.2f} - {'PASS' if metrics['iv30_rv30_pass'] else 'FAIL'}")
                logger.info(
                    f"   Slope: {metrics['ts_slope_0_45']:.6f} - {'PASS' if metrics['ts_slope_pass'] else 'FAIL'}")
            else:
                logger.error(f"❌ Failed to analyze metrics for {symbol}")

        # Print final results
        logger.info("\n=== Test Summary ===")
        logger.info(f"Successfully analyzed {len(results)} of {len(test_symbols)} symbols")

        passed_symbols = [s for s, r in results.items() if r.get('all_criteria_met', False)]
        if passed_symbols:
            logger.info(f"Symbols that passed screening criteria: {', '.join(passed_symbols)}")
        else:
            logger.info("No symbols passed all screening criteria")

        return True

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False

    finally:
        # Disconnect from IBKR
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IBKR")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)