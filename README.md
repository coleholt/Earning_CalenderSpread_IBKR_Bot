# Earnings Calendar Spread Bot

An automated options trading bot that implements a calendar spread strategy around earnings announcements, specifically designed for AWS deployment with Interactive Brokers.

## Overview

This bot automatically trades calendar spreads on stocks with upcoming earnings announcements. It follows a specific strategy of entering positions shortly before market close and exiting after the next market open. The bot is optimized to work with free API tiers and includes comprehensive monitoring, reporting, and error handling features.

![Calendar Spread Strategy](https://i.imgur.com/example.png)

## Strategy Details

### Key Features

- **Timing**: Enters positions 15 minutes before market close, exits 15 minutes after the next market open
- **Target Selection**: Trades stocks with earnings announcements after market close (AMC) today or before market open (BMO) tomorrow
- **Strategy**: Calendar spreads using either calls or puts (whichever is cheaper)
  - Short leg: 1-7 trading days to expiration (with 1 DTE meaning expiring on exit day, prioritizing closest to 1 DTE)
  - Long leg: 21-45 trading days to expiration (prioritizing closest to 30 trading DTE)
- **Risk Management**: Limits each trade to 6.5% of total account value
- **Screening Criteria**:
  - Term structure slope ≤ -0.00406 (negative slope is preferred)
  - 30-day average volume ≥ 1,500,000
  - Pre-earnings IV/RV ratio ≥ 1.25
- **Safety Features**: Handles early assignment by automatically closing positions

## Components

The bot consists of several key components:

1. **Main Bot Script** (`earnings_calendar_bot.py`): Core trading logic, screening, and execution
2. **Watchdog Monitor** (`earnings_bot_watchdog.py`): Monitors bot health and sends alerts
3. **Email Test Script** (`test_email_config.py`): Tests email notification setup
4. **IBKR Connection Test** (`test_ibkr.py`): Verifies IB connection works properly
5. **Update Reminder** (`update_reminder.py`): Sends regular reminders to update the system
6. **Cron Setup Script** (`setup_cron.sh`): Automates scheduling of all components

## Optimized for Free API Tiers

The bot is designed to work with free API tiers:

### API-Efficient Design

- **Early Pre-Screening**: Performs analysis in the morning to avoid API rate limits
- **Caching System**: Stores pre-screened data to minimize API calls
- **YFinance Integration**: Uses free YFinance data when possible
- **Rate Limiting**: Built-in delays between API calls to respect limitations

### Using the Pre-Screening Feature

To use the pre-screening mode (recommended for free API tiers):

```bash
# Run pre-screening in the morning
python earnings_calendar_bot.py prescreen

# Later in the day, run the normal bot which will use pre-screened data
python earnings_calendar_bot.py
```

## Monitoring and Reporting Features

The bot includes comprehensive monitoring, reporting, and error handling:

### Email Notifications

- **Error Alerts**: Automated email alerts are sent when errors occur
- **Performance Reports**: Daily reports showing P&L for various timeframes
- **Position Summaries**: Current positions and their status

### Watchdog System

- **Process Monitoring**: Continuously tracks if the bot is running
- **Heartbeat Mechanism**: Detects if the bot has frozen
- **Auto-restart Capability**: Optional feature to restart the bot if it fails

### Performance Tracking

The bot maintains a comprehensive trade history with:

- Yesterday's P&L
- Last 7 days performance
- Last 30 days performance
- Last 90 days performance
- Last year performance
- Last 2 years performance
- Last 5 years performance (if applicable)

## AWS Deployment

This bot is specifically designed to run on an AWS EC2 instance:

- **Deployment Guide**: See `AWS_IBKR_Setup_Instructions.md` for step-by-step setup instructions
- **Minimal Resources**: Works with t2.micro instances (free tier eligible)
- **Persistent Execution**: Includes monitoring to ensure continuous operation
- **Security Focused**: Follows AWS best practices for security

## Prerequisites

- Interactive Brokers account with paper trading enabled
- IBKR Trader Workstation (TWS) or IB Gateway installed
- AWS account with EC2 instance (Ubuntu recommended)
- A Finnhub API key (free tier is sufficient)
- Email account for notifications (Gmail recommended)

## Installation and Setup

See the detailed installation guide in `AWS_IBKR_Setup_Instructions.md`

## Usage

The bot supports multiple execution modes:

```bash
# Morning pre-screening (8:00 AM recommended)
python earnings_calendar_bot.py prescreen

# Normal execution (3:00 PM recommended)
python earnings_calendar_bot.py

# Immediate execution (skips waiting)
python earnings_calendar_bot.py run

# Test email configuration
python test_email_config.py

# Test IBKR connection
python test_ibkr.py

# Setup automated execution
./setup_cron.sh
```

## Configuration

Edit the `config.json` file to customize the bot's behavior:

```json
{
    "strategy": {
        "entry_minutes_before_close": 15,
        "exit_minutes_after_open": 15,
        "screening_criteria": {
            "min_avg_volume": 1500000,
            "min_iv30_rv30_ratio": 1.25,
            "max_term_structure_slope": -0.00406
        }
    },
    "email": {
        "enabled": true,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "YOUR_EMAIL@gmail.com",
        "sender_password": "YOUR_APP_PASSWORD",
        "recipient_email": "holt.maven@gmail.com"
    }
}
```

## Maintenance

### Regular Updates

When you receive an update reminder email, follow these steps:

1. Connect to your AWS instance
2. Navigate to the bot directory and activate the environment
3. Update dependencies
4. Test the connection
5. Mark as updated: `python update_reminder.py --mark-updated`

## Disclaimer

Trading options, especially around earnings announcements, involves substantial risk. This bot is provided for educational purposes only. Use at your own risk and with appropriate risk management. Always test with paper trading before deploying with real funds.

## License

[MIT License](LICENSE)