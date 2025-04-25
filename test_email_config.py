#!/usr/bin/env python3
"""
Test script for email configuration.
Run this script to test if your email settings are correctly configured.
"""

import json
import logging
import smtplib
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_email_config():
    """Load email configuration from config file"""
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            email_config = config.get('email', {})

            email_settings = {
                'enabled': email_config.get('enabled', True),
                'smtp_server': email_config.get('smtp_server', 'smtp.gmail.com'),
                'smtp_port': email_config.get('smtp_port', 587),
                'sender_email': email_config.get('sender_email', ''),
                'sender_password': email_config.get('sender_password', ''),
                'recipient_email': email_config.get('recipient_email', 'holt.maven@gmail.com'),
                'notification_level': email_config.get('notification_level', 'ERROR')
            }

            return email_settings
    except Exception as e:
        logger.error(f"Failed to load email config: {e}")
        return None


def test_email_connection():
    """Test SMTP connection and login"""
    email_config = load_email_config()

    if not email_config:
        logger.error("Failed to load email configuration")
        return False

    if not email_config['enabled']:
        logger.warning("Email notifications are disabled in the config")
        return False

    if not email_config['sender_email'] or not email_config['sender_password']:
        logger.error("Email credentials not configured. Please add sender_email and sender_password to config.json")
        return False

    try:
        logger.info(f"Testing connection to {email_config['smtp_server']}:{email_config['smtp_port']}...")
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()

        logger.info(f"Attempting login for {email_config['sender_email']}...")
        server.login(email_config['sender_email'], email_config['sender_password'])

        logger.info("Login successful!")
        server.quit()

        return True
    except Exception as e:
        logger.error(f"Email connection test failed: {e}")
        return False


def send_test_email():
    """Send a test email"""
    email_config = load_email_config()

    if not email_config:
        logger.error("Failed to load email configuration")
        return False

    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Earnings Bot - Email Test"
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']

        # Add plain text part
        text_content = f"""
This is a test email from your Earnings Calendar Bot.

If you're receiving this message, your email configuration is working correctly!

Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
- SMTP Server: {email_config['smtp_server']}
- SMTP Port: {email_config['smtp_port']}
- Sender: {email_config['sender_email']}
- Recipient: {email_config['recipient_email']}
- Notification Level: {email_config['notification_level']}

You can now receive automated alerts and performance reports from the bot.
"""
        text_part = MIMEText(text_content, 'plain')
        msg.attach(text_part)

        # Add HTML part
        html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333366; }}
        .success {{ color: green; font-weight: bold; }}
        .config {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Earnings Bot - Email Test</h1>

    <p class="success">If you're receiving this message, your email configuration is working correctly!</p>

    <p>Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="config">
        <h3>Configuration:</h3>
        <ul>
            <li><strong>SMTP Server:</strong> {email_config['smtp_server']}</li>
            <li><strong>SMTP Port:</strong> {email_config['smtp_port']}</li>
            <li><strong>Sender:</strong> {email_config['sender_email']}</li>
            <li><strong>Recipient:</strong> {email_config['recipient_email']}</li>
            <li><strong>Notification Level:</strong> {email_config['notification_level']}</li>
        </ul>
    </div>

    <p>You can now receive automated alerts and performance reports from the bot.</p>
</body>
</html>
"""
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        # Connect to server
        logger.info(f"Connecting to {email_config['smtp_server']}:{email_config['smtp_port']}...")
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()

        logger.info(f"Logging in as {email_config['sender_email']}...")
        server.login(email_config['sender_email'], email_config['sender_password'])

        # Send email
        logger.info(f"Sending test email to {email_config['recipient_email']}...")
        server.send_message(msg)
        server.quit()

        logger.info("Test email sent successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to send test email: {e}")
        return False


def main():
    """Main function"""
    logger.info("Testing email configuration...")

    # Test connection first
    if test_email_connection():
        logger.info("Email server connection test passed!")

        # Then send a test email
        if send_test_email():
            logger.info("Test email sent successfully!")
            logger.info(f"Please check your inbox at {load_email_config()['recipient_email']}")
            return True
        else:
            logger.error("Failed to send test email.")
            return False
    else:
        logger.error("Email server connection test failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)