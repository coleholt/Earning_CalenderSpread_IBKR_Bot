#!/usr/bin/env python3
"""
Update Reminder Script

This script sends periodic reminders to update the earnings calendar bot
and check for new versions or updates to dependencies.

Usage:
    python update_reminder.py
"""

import json
import logging
import datetime
import smtplib
import os
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/update_reminder.log'),
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


def get_update_info():
    """Collect information about installed packages and last update"""
    try:
        # Get pip packages information
        import subprocess
        result = subprocess.run(['pip', 'list', '--outdated'],
                                capture_output=True, text=True, check=True)
        outdated_packages = result.stdout

        # Get system information
        import platform
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'node': platform.node(),
            'processor': platform.processor()
        }

        # Get last update date from tracking file
        last_update_date = "Never"
        if os.path.exists('last_update.txt'):
            with open('last_update.txt', 'r') as f:
                last_update_date = f.read().strip()

        # Calculate days since last update
        try:
            last_date = datetime.datetime.strptime(last_update_date, '%Y-%m-%d').date()
            days_since_update = (datetime.date.today() - last_date).days
        except Exception:
            days_since_update = 999  # A large number if we can't calculate

        return {
            'outdated_packages': outdated_packages,
            'system_info': system_info,
            'last_update_date': last_update_date,
            'days_since_update': days_since_update
        }
    except Exception as e:
        logger.error(f"Error getting update info: {e}")
        return {
            'outdated_packages': 'Error fetching package information',
            'system_info': {},
            'last_update_date': 'Unknown',
            'days_since_update': 999
        }


def send_reminder_email(config, update_info):
    """Send reminder email"""
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
        msg['Subject'] = f"Earnings Bot: Update Reminder"
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']

        # Format text content
        text_content = f"""
EARNINGS CALENDAR BOT - UPDATE REMINDER
=======================================

It's been {update_info['days_since_update']} days since your last update check.

Last update date: {update_info['last_update_date']}
Current date: {datetime.date.today().strftime('%Y-%m-%d')}

System Information:
- Platform: {update_info['system_info'].get('platform', 'Unknown')}
- Python: {update_info['system_info'].get('python_version', 'Unknown')}

Recommended Update Actions:
1. Pull the latest code: `git pull origin main`
2. Update dependencies: `pip install --upgrade -r requirements.txt`
3. Check configuration changes
4. Test the connection: `python test_ibkr.py`
5. Record your update: `date +%Y-%m-%d > last_update.txt`

Outdated Packages:
{update_info['outdated_packages']}

Remember to also check:
- IBKR software updates
- API key renewals
- AWS instance maintenance

This is an automated reminder. Updates help ensure the bot continues to function properly.
        """
        text_part = MIMEText(text_content, 'plain')
        msg.attach(text_part)

        # Format HTML content
        html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333366; }}
        .warning {{ color: #cc6600; font-weight: bold; }}
        .code {{ background-color: #f0f0f0; padding: 10px; font-family: monospace; border-radius: 5px; }}
        .steps {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #333366; }}
        .packages {{ max-height: 200px; overflow-y: auto; font-family: monospace; }}
    </style>
</head>
<body>
    <h1>Earnings Calendar Bot - Update Reminder</h1>

    <p class="warning">It's been {update_info['days_since_update']} days since your last update check.</p>

    <p>Last update date: <strong>{update_info['last_update_date']}</strong><br>
    Current date: <strong>{datetime.date.today().strftime('%Y-%m-%d')}</strong></p>

    <h2>System Information</h2>
    <ul>
        <li><strong>Platform:</strong> {update_info['system_info'].get('platform', 'Unknown')}</li>
        <li><strong>Python:</strong> {update_info['system_info'].get('python_version', 'Unknown')}</li>
    </ul>

    <h2>Recommended Update Actions</h2>
    <div class="steps">
        <ol>
            <li>Pull the latest code: <div class="code">git pull origin main</div></li>
            <li>Update dependencies: <div class="code">pip install --upgrade -r requirements.txt</div></li>
            <li>Check for configuration changes</li>
            <li>Test the connection: <div class="code">python test_ibkr.py</div></li>
            <li>Record your update: <div class="code">date +%Y-%m-%d > last_update.txt</div></li>
        </ol>
    </div>

    <h2>Outdated Packages</h2>
    <div class="packages">
        <pre>{update_info['outdated_packages']}</pre>
    </div>

    <h2>Additional Reminders</h2>
    <ul>
        <li>Check for IBKR software updates</li>
        <li>Verify API key renewals</li>
        <li>Perform AWS instance maintenance</li>
    </ul>

    <p><em>This is an automated reminder. Regular updates help ensure the bot continues to function properly.</em></p>
</body>
</html>
        """
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        # Connect to server
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])

        # Send email
        server.send_message(msg)
        server.quit()

        logger.info(f"Update reminder email sent to {email_config['recipient_email']}")
        return True

    except Exception as e:
        logger.error(f"Failed to send update reminder email: {e}")
        return False


def mark_update_complete():
    """Mark update as complete by saving today's date"""
    try:
        with open('last_update.txt', 'w') as f:
            f.write(datetime.date.today().strftime('%Y-%m-%d'))
        logger.info("Update marked as complete")
        return True
    except Exception as e:
        logger.error(f"Error marking update as complete: {e}")
        return False


def should_send_reminder():
    """Determine if it's time to send a reminder"""
    try:
        # Get update frequency from config
        config = load_config()
        update_frequency = config.get('monitoring', {}).get('update_reminder_frequency_days', 30)

        # Check last update date
        if os.path.exists('last_update.txt'):
            with open('last_update.txt', 'r') as f:
                last_update_str = f.read().strip()
                try:
                    last_update = datetime.datetime.strptime(last_update_str, '%Y-%m-%d').date()
                    days_since = (datetime.date.today() - last_update).days
                    return days_since >= update_frequency
                except ValueError:
                    # Invalid date format, should send reminder
                    return True
        else:
            # No update record, should send reminder
            return True
    except Exception as e:
        logger.error(f"Error checking if reminder should be sent: {e}")
        # If there's an error, better to send a reminder just in case
        return True


def main():
    """Main function"""
    try:
        # Check if we need to send a reminder
        if not should_send_reminder():
            logger.info("No update reminder needed at this time")
            return True

        # If update flag is provided, mark as updated and exit
        if len(sys.argv) > 1 and sys.argv[1] == "--mark-updated":
            return mark_update_complete()

        logger.info("Sending update reminder email")
        config = load_config()
        update_info = get_update_info()

        # Send the reminder email
        success = send_reminder_email(config, update_info)

        logger.info(f"Reminder email {'sent successfully' if success else 'failed to send'}")
        return success

    except Exception as e:
        logger.error(f"Error in update reminder: {e}")
        return False


if __name__ == "__main__":
    main()