#!/usr/bin/env python3
"""
Test script to verify connection to Interactive Brokers
Run this script to ensure your IBKR connection is working properly
"""

from ib_insync import *
import sys
import json
import logging
import os
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config file"""
    try:
        # Find config file in current directory
        if os.path.exists('config.json'):
            config_path = 'config.json'
        else:
            # Look in parent directory if not found
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(parent_dir, 'config.json')
            if not os.path.exists(config_path):
                logger.error("Config file not found")
                return None

        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def test_connection():
    """Test connection to IBKR"""
    try:
        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration. Using default connection params.")
            host = '127.0.0.1'
            port = 7497
            client_id = 1
        else:
            # Extract connection parameters from config
            ibkr_config = config.get('ibkr', {})
            host = ibkr_config.get('host', '127.0.0.1')
            port = ibkr_config.get('port', 7497)
            client_id = ibkr_config.get('client_id', 1)

        logger.info(f"Connecting to IBKR at {host}:{port} with client ID {client_id}")

        # Initialize connection
        ib = IB()

        # Connect to IBKR
        ib.connect(host, port, clientId=client_id)

        # Check if connected
        if ib.isConnected():
            logger.info("✅ Successfully connected to IBKR API")
            logger.info(f"Server version: {ib.client.serverVersion()}")
            logger.info(f"Server time: {ib.reqCurrentTime()}")

            # Get account information
            account_summary = ib.accountSummary()
            account = account_summary[0].account if account_summary else "Unknown"
            logger.info(f"Account: {account}")

            # Get net liquidation value
            net_liq = next((float(item.value) for item in account_summary if item.tag == 'NetLiquidation'), 0)
            logger.info(f"Net Liquidation Value: ${net_liq:.2f}")

            # Test market data request
            spy = Stock('SPY', 'SMART', 'USD')
            ib.qualifyContracts(spy)

            # Request market data
            logger.info("Requesting market data for SPY...")
            ticker = ib.reqMktData(spy)

            # Wait for data to arrive
            timeout = 10
            start_time = time.time()
            while time.time() - start_time < timeout:
                ib.sleep(1)
                if ticker.marketPrice() > 0:
                    break

            # Check if we got data
            price = ticker.marketPrice()
            if price > 0:
                logger.info(f"✅ Successfully received market data: SPY price = ${price:.2f}")
            else:
                logger.warning("⚠️ Couldn't get market data in time, but connection is working")

            # Test option chain request
            logger.info("Requesting option chain data...")
            chains = ib.reqSecDefOptParams(spy.symbol, '', spy.secType, spy.conId)

            if chains:
                logger.info(f"✅ Successfully received option chain data: {len(chains)} chains available")
                chain = next((c for c in chains if c.exchange == 'SMART'), None)
                if chain:
                    logger.info(f"   Expirations: {len(chain.expirations)} available")
                    logger.info(f"   Strikes: {len(chain.strikes)} available")
            else:
                logger.warning("⚠️ Couldn't get option chain data, but connection is working")

            # Disconnect
            ib.disconnect()
            logger.info("Disconnected from IBKR API")

            return True
        else:
            logger.error("❌ Failed to connect to IBKR API")
            return False

    except Exception as e:
        logger.error(f"❌ Error connecting to IBKR: {e}")
        return False


def test_paper_trading():
    """Verify if connected to paper trading account"""
    try:
        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration. Using default connection params.")
            host = '127.0.0.1'
            port = 7497
            client_id = 1
        else:
            # Extract connection parameters from config
            ibkr_config = config.get('ibkr', {})
            host = ibkr_config.get('host', '127.0.0.1')
            port = ibkr_config.get('port', 7497)
            client_id = ibkr_config.get('client_id', 1)

        # Initialize connection
        ib = IB()

        # Connect to IBKR
        ib.connect(host, port, clientId=client_id)

        if ib.isConnected():
            # Check if this is a paper account
            account_summary = ib.accountSummary()
            account_type = next((item.value for item in account_summary if item.tag == 'AccountType'), None)

            if account_type:
                logger.info(f"Account type: {account_type}")
                if 'PAPER' in account_type.upper():
                    logger.info("✅ Connected to PAPER trading account")
                    is_paper = True
                else:
                    logger.warning("⚠️ WARNING: Connected to LIVE trading account!")
                    is_paper = False
            else:
                logger.warning("⚠️ Could not determine account type")
                is_paper = None

            # Disconnect
            ib.disconnect()

            return is_paper
        else:
            logger.error("❌ Failed to connect to IBKR API")
            return None

    except Exception as e:
        logger.error(f"❌ Error checking account type: {e}")
        return None


def run_all_tests():
    """Run all tests and report results"""
    logger.info("=== IBKR Connection Test ===")

    # Test basic connection
    logger.info("\n== Testing basic connection ==")
    conn_success = test_connection()

    # Test if paper trading
    logger.info("\n== Checking account type ==")
    is_paper = test_paper_trading()

    # Print summary
    logger.info("\n=== Test Summary ===")
    if conn_success:
        logger.info("✅ Connection test: PASSED")
    else:
        logger.info("❌ Connection test: FAILED")

    if is_paper is True:
        logger.info("✅ Paper trading check: PASSED (Paper account confirmed)")
    elif is_paper is False:
        logger.info("⚠️ Paper trading check: WARNING (Live account detected!)")
    else:
        logger.info("❓ Paper trading check: UNKNOWN")

    if conn_success and is_paper is True:
        logger.info("\n✅ All tests PASSED! Your bot should work correctly.")
        return True
    elif conn_success and is_paper is False:
        logger.info("\n⚠️ Connection works but using LIVE account! Switch to paper trading for testing.")
        return False
    else:
        logger.info("\n❌ One or more tests FAILED. Please check the logs and your configuration.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)