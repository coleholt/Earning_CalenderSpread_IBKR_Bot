#!/usr/bin/env python3
"""
Watchdog script for earnings calendar bot monitoring.
Run this script via cron job to check if the main bot is running.

Example cron entry (check every 30 minutes):
*/30 * * * * /usr/bin/python3 /path/to/earnings_bot_watchdog.py

If the bot has stopped running, this script will send an email alert.
"""

import os
import json
import datetime
import logging
import smtplib
import sys
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('earnings_bot_watchdog.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config file"""
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def send_email_alert(subject, message, config):
    """Send email alert"""
    if not config or 'email' not in config:
        logger.error("Email config not available")
        return False

    email_config = config['email']

    if not email_config.get('enabled', False):
        logger.warning("Email notifications disabled")
        return False

    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Earnings Bot Watchdog Alert: {subject}"
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']

        # Add plain text part
        text_part = MIMEText(message, 'plain')
        msg.attach(text_part)

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


def check_bot_status():
    """Check if the earnings calendar bot is running"""
    watchdog_file = "watchdog.json"

    try:
        if not os.path.exists(watchdog_file):
            logger.error("Watchdog file not found. Bot may not be running.")
            return False, "Watchdog file not found. Bot may have stopped running or never started."

        with open(watchdog_file, 'r') as f:
            watchdog_data = json.load(f)

        last_heartbeat = datetime.datetime.fromisoformat(watchdog_data["last_heartbeat"])
        now = datetime.datetime.now()

        config = load_config()
        check_interval = config.get('monitoring', {}).get('watchdog_check_interval_minutes', 10)

        # Check if heartbeat is more than configured interval old
        if (now - last_heartbeat).total_seconds() > (check_interval * 60):
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
                message = f"Bot process (PID {pid}) appears to be running but heartbeat is stale. Last update was {last_heartbeat.isoformat()}."
            else:
                message = f"Bot process (PID {pid}) is not running. Bot has crashed or been terminated. Last heartbeat was {last_heartbeat.isoformat()}."

            return False, message

        # Bot is running normally
        return True, f"Bot is running normally. Last heartbeat: {last_heartbeat.isoformat()}"

    except Exception as e:
        error_msg = f"Error checking watchdog status: {e}"
        logger.error(error_msg)
        return False, error_msg


def try_restart_bot():
    """Attempt to restart the bot if it's not running"""
    try:
        bot_script = "earnings_calendar_bot.py"

        # Check if script exists
        if not os.path.exists(bot_script):
            logger.error(f"Bot script not found at {bot_script}")
            return False, f"Bot script not found at {bot_script}"

        # Start the bot as a background process
        subprocess.Popen(
            [sys.executable, bot_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        logger.info("Bot restart attempted")
        return True, "Bot restart attempted"

    except Exception as e:
        error_msg = f"Error trying to restart bot: {e}"
        logger.error(error_msg)
        return False, error_msg


def main():
    """Main watchdog function"""
    logger.info("Earnings bot watchdog check started")

    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return

    # Check if bot is running
    bot_status, status_message = check_bot_status()

    if not bot_status:
        logger.warning(f"Bot check failed: {status_message}")

        # Try to restart the bot if enabled in config
        auto_restart = config.get('monitoring', {}).get('auto_restart', False)

        if auto_restart:
            restart_success, restart_message = try_restart_bot()

            if restart_success:
                alert_message = f"""
Bot was not running properly and watchdog attempted a restart.

Status: {status_message}
Restart: {restart_message}

Please check the logs for more information.
"""
            else:
                alert_message = f"""
Bot was not running properly and watchdog FAILED to restart it.

Status: {status_message}
Restart attempt: {restart_message}

MANUAL INTERVENTION REQUIRED.

Please check the logs and restart the bot manually.
"""
        else:
            alert_message = f"""
Bot is not running properly and automatic restart is disabled.

Status: {status_message}

MANUAL INTERVENTION REQUIRED.

Please check the logs and restart the bot manually.
"""

        # Send email alert
        send_email_alert("Bot Not Running", alert_message, config)
    else:
        logger.info(f"Bot check succeeded: {status_message}")


if __name__ == "__main__":
    main()