# Setting Up the Earnings Calendar Bot on AWS with IBKR Paper Trading

This guide provides detailed instructions for deploying the earnings calendar bot on an AWS EC2 instance and connecting it to an Interactive Brokers paper trading account.

## Prerequisites

1. An AWS account with an EC2 instance running (Ubuntu recommended)
2. Interactive Brokers account with paper trading enabled
3. IBKR Trader Workstation (TWS) or IB Gateway installed
4. Basic familiarity with Linux and SSH
5. A Finnhub API key (free tier is sufficient)
6. An email account for notifications (Gmail recommended)

## Step 1: Set Up Your IBKR Paper Trading Environment

### 1.1. Install TWS or IB Gateway

First, install either TWS or IB Gateway on your local machine for initial setup:

1. Log in to your IBKR account
2. Download and install TWS or IB Gateway from the IBKR website
3. Configure it to connect using your paper trading credentials

### 1.2. Enable API Access

1. Launch TWS or IB Gateway
2. Go to File > Global Configuration > API > Settings
3. Check "Enable ActiveX and Socket Clients"
4. Set "Socket port" to 7497 (default for paper trading)
5. Check "Allow connections from localhost only" (we'll modify this later)
6. Uncheck "Read-Only API" to allow trading
7. Set "Master API client ID" to 0

### 1.3. Prepare for Remote Access

Since we'll be accessing the API from AWS:
1. Uncheck "Allow connections from localhost only"
2. Add your AWS instance's IP address to the "Trusted IPs" list
3. Apply changes and restart TWS/IB Gateway

## Step 2: Prepare AWS Instance

### 2.1. Launch EC2 Instance

1. Log in to AWS Console
2. Launch a t2.micro (or larger) instance with Ubuntu Server (22.04 LTS recommended)
3. Configure security groups to allow SSH (port 22) from your IP
4. Create and download your key pair (.pem file)
5. Launch the instance

### 2.2. Connect to Your Instance

```bash
chmod 400 your-key-pair.pem
ssh -i your-key-pair.pem ubuntu@your-ec2-public-dns
```

### 2.3. Install Required Packages

```bash
# Update and install system packages
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-venv git

# Create a virtual environment for the bot
mkdir -p ~/earnings_bot/logs
cd ~/earnings_bot
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install ib_insync finnhub-python pandas numpy requests scipy pytz croniter yfinance matplotlib
```

## Step 3: Set Up the Bot Files

### 3.1. Clone the Repository or Create Files Manually

**Option 1: Clone from repository (if available)**
```bash
git clone https://github.com/yourusername/earnings-calendar-bot.git .
```

**Option 2: Create files manually**
Create the following files:
- `earnings_calendar_bot.py`
- `earnings_bot_watchdog.py`
- `test_email_config.py`
- `update_reminder.py`
- `setup_cron.sh`
- `test_ibkr.py`

### 3.2. Create Configuration File

Create `config.json`:

```bash
cat > config.json << 'EOF'
{
    "api_keys": {
        "finnhub": "YOUR_FINNHUB_API_KEY"
    },
    "ibkr": {
        "host": "YOUR_LOCAL_IP_OR_HOSTNAME",
        "port": 7497,
        "client_id": 1,
        "paper_trading": true
    },
    "account": {
        "size": 10000,
        "risk_per_trade": 6.5
    },
    "strategy": {
        "entry_minutes_before_close": 15,
        "exit_minutes_after_open": 15,
        "short_dte_min": 1,
        "short_dte_max": 7,
        "long_dte_min": 21,
        "long_dte_max": 45,
        "long_dte_target": 30,
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
        "recipient_email": "holt.maven@gmail.com",
        "notification_level": "ERROR",
        "daily_report_time": "20:00"
    },
    "monitoring": {
        "check_interval_seconds": 300,
        "watchdog_check_interval_minutes": 10,
        "auto_restart": true,
        "update_reminder_frequency_days": 30
    },
    "logging": {
        "log_file": "logs/earnings_calendar_bot.log",
        "log_level": "INFO"
    }
}
EOF
```

**Important:** 
- Update `ibkr.host` with the IP address or hostname where TWS/IB Gateway is running
- Update `api_keys.finnhub` with your Finnhub API key
- Update email settings with your Gmail address and app password

### 3.3. Make Scripts Executable

```bash
chmod +x earnings_calendar_bot.py
chmod +x earnings_bot_watchdog.py
chmod +x test_email_config.py
chmod +x update_reminder.py
chmod +x setup_cron.sh
chmod +x test_ibkr.py
```

## Step 4: Configure TWS/IB Gateway for Remote Access

### 4.1. Set Up TWS/IB Gateway for Automatic Login

To allow the gateway to run headlessly, configure it for auto-login:
1. Launch TWS/IB Gateway
2. Go to File > Global Configuration > API > Settings
3. Check "Enable API" and "Allow connections from any IP"
4. Check "Trusted IPs only" and add your AWS instance's IP
5. Go to Configure > Settings > API > Precautions
6. Set "Read-Only API" to "Disabled"
7. Set "Auto restart API" to "Enabled"
8. Set "API log level" to "Detail"

### 4.2. Configure TWS/IB Gateway to Run at Startup

**For Windows:**
1. Create a shortcut to IBGateway.exe
2. Add command-line parameters: `/autostartapi /user YOUR_USERNAME /password YOUR_PASSWORD /papertrading`
3. Place the shortcut in your startup folder

**For Linux:**
```bash
# Create a startup script
cat > ~/ibgateway-start.sh << 'EOF'
#!/bin/bash
cd /path/to/ibgateway/installation
./ibgateway username=YOUR_USERNAME password=YOUR_PASSWORD -api
EOF
chmod +x ~/ibgateway-start.sh

# Add to crontab
crontab -e
```

Add this line to your crontab:
```
@reboot /home/username/ibgateway-start.sh
```

## Step 5: Test the Connection

### 5.1. Test IBKR Connection

Update `YOUR_TWS_HOST` in the test script with your TWS/IB Gateway IP address:

```bash
python3 test_ibkr.py
```

You should see successful connection information.

### 5.2. Test Email Configuration

```bash
python3 test_email_config.py
```

Check if you receive the test email.

## Step 6: Set Up Update Reminders

### 6.1. Initialize Update Tracking

Mark today as your first update:

```bash
date +%Y-%m-%d > last_update.txt
```

### 6.2. Test Update Reminder

```bash
python3 update_reminder.py
```

You should receive an email reminding you about updates.

## Step 7: Configure Scheduled Execution

### 7.1. Set Up Cron Jobs

Run the setup script to configure cron jobs:

```bash
./setup_cron.sh
```

This will create jobs for:
- Pre-screening at 8:00 AM on weekdays
- Main bot execution at 3:00 PM on weekdays
- Watchdog checks every 30 minutes during market hours
- Update reminders once a month

### 7.2. Verify Cron Setup

```bash
crontab -l
```

You should see entries for each of the scheduled tasks.

## Step 8: Monitor and Maintain

### 8.1. Monitor Logs

```bash
# View bot logs
tail -f logs/earnings_calendar_bot.log

# View cron job logs
tail -f cron_prescreen.log
tail -f cron_bot.log
tail -f cron_watchdog.log
```

### 8.2. Run a Manual Test

```bash
# Run pre-screening manually
python3 earnings_calendar_bot.py prescreen

# Run the bot in immediate mode
python3 earnings_calendar_bot.py run
```

### 8.3. Setup Persistent Execution with Screen

To keep the bot running even after you disconnect from SSH:

```bash
sudo apt install screen
screen -S earnings_bot
cd ~/earnings_bot
source venv/bin/activate
python3 earnings_calendar_bot.py

# Press Ctrl+A, then D to detach
# To reconnect later:
screen -r earnings_bot
```

## Step 9: Maintenance and Updates

### 9.1. Regular Updates

When you receive an update reminder email, follow these steps:

1. Connect to your AWS instance
2. Navigate to the bot directory and activate the environment:
   ```bash
   cd ~/earnings_bot
   source venv/bin/activate
   ```
3. If using Git, pull the latest code:
   ```bash
   git pull origin main
   ```
4. Update dependencies:
   ```bash
   pip install --upgrade ib_insync finnhub-python pandas numpy
   ```
5. Test the connection:
   ```bash
   python3 test_ibkr.py
   ```
6. Mark as updated:
   ```bash
   python3 update_reminder.py --mark-updated
   ```

### 9.2. System Updates

Regularly update the AWS system:

```bash
sudo apt update && sudo apt upgrade -y
```

### 9.3. API Key Renewals

Check for API key expirations:
- Finnhub API keys may need to be renewed
- Email app passwords may expire

## AWS-Specific Considerations

### Keep Instance Running

1. Go to EC2 Dashboard
2. Select your instance
3. Actions > Instance Settings > Change Shutdown Behavior
4. Select "Stop" instead of "Terminate"
5. Enable termination protection

### Set Up CloudWatch Alarms

Consider setting up alarms for:
- Instance status
- CPU utilization
- Memory usage

### Consider Using Elastic IP

Assign an Elastic IP to your instance to keep its public IP address consistent, which is important for IBKR API connection rules.

## Security Considerations

### Firewall Configuration

Configure your EC2 security group to allow:
- SSH (port 22) from your IP only
- Connection to IBKR (outbound to port 7497)

### Secure Config Files

```bash
chmod 600 config.json
```

## Troubleshooting

### Common Issues

1. **Connection to IBKR fails**
   - Check if TWS/IB Gateway is running
   - Verify IP address is in trusted IPs
   - Ensure correct port (7497 for paper trading)

2. **Email notifications not sending**
   - Check SMTP settings
   - Verify app password is correct
   - Ensure email is enabled in config

3. **Bot not executing trades**
   - Check logs for errors
   - Verify paper trading account has sufficient funds
   - Ensure API permissions allow trading

4. **AWS instance disconnects**
   - Use `screen` or `tmux` to maintain sessions
   - Check instance type meets memory requirements

For additional support, refer to the logs or contact the developer.