#!/bin/bash

# Script to set up cron jobs for the earnings calendar bot
# This automates pre-screening in the morning and regular execution before market close

# Get the absolute path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BOT_SCRIPT="${SCRIPT_DIR}/earnings_calendar_bot.py"
WATCHDOG_SCRIPT="${SCRIPT_DIR}/earnings_bot_watchdog.py"
REMINDER_SCRIPT="${SCRIPT_DIR}/update_reminder.py"

# Check if the scripts exist
if [ ! -f "$BOT_SCRIPT" ]; then
    echo "Error: Bot script not found at $BOT_SCRIPT"
    exit 1
fi

if [ ! -f "$WATCHDOG_SCRIPT" ]; then
    echo "Error: Watchdog script not found at $WATCHDOG_SCRIPT"
    exit 1
fi

# Make sure the scripts are executable
chmod +x "$BOT_SCRIPT"
chmod +x "$WATCHDOG_SCRIPT"
if [ -f "$REMINDER_SCRIPT" ]; then
    chmod +x "$REMINDER_SCRIPT"
fi

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# Create a temporary file for the cron jobs
TEMP_CRON=$(mktemp)

# Export current cron jobs
crontab -l > "$TEMP_CRON" 2>/dev/null || echo "# New crontab" > "$TEMP_CRON"

# Add comment
echo "" >> "$TEMP_CRON"
echo "# Earnings Calendar Bot Cron Jobs" >> "$TEMP_CRON"

# Add cron jobs
# 1. Pre-screening at 8:00 AM every weekday (Monday-Friday)
echo "0 8 * * 1-5 cd $SCRIPT_DIR && /usr/bin/python3 $BOT_SCRIPT prescreen >> ${SCRIPT_DIR}/logs/cron_prescreen.log 2>&1" >> "$TEMP_CRON"

# 2. Main bot execution at 3:00 PM every weekday (to prepare for 15 min before close)
echo "0 15 * * 1-5 cd $SCRIPT_DIR && /usr/bin/python3 $BOT_SCRIPT >> ${SCRIPT_DIR}/logs/cron_bot.log 2>&1" >> "$TEMP_CRON"

# 3. Watchdog check every 30 minutes during market hours
echo "*/30 8-16 * * 1-5 cd $SCRIPT_DIR && /usr/bin/python3 $WATCHDOG_SCRIPT >> ${SCRIPT_DIR}/logs/cron_watchdog.log 2>&1" >> "$TEMP_CRON"

# 4. Monthly update reminder (1st day of each month at 9:00 AM)
if [ -f "$REMINDER_SCRIPT" ]; then
    echo "0 9 1 * * cd $SCRIPT_DIR && /usr/bin/python3 $REMINDER_SCRIPT >> ${SCRIPT_DIR}/logs/cron_reminder.log 2>&1" >> "$TEMP_CRON"
fi

# Install the cron jobs
crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo "Cron jobs have been set up successfully:"
echo "1. Pre-screening at 8:00 AM every weekday (Monday-Friday)"
echo "2. Main bot execution at 3:00 PM every weekday"
echo "3. Watchdog check every 30 minutes during market hours (8 AM - 4 PM)"
if [ -f "$REMINDER_SCRIPT" ]; then
    echo "4. Monthly update reminder on the 1st of each month at 9:00 AM"
fi
echo ""
echo "You can verify the cron jobs with 'crontab -l'"

# Initialize the last_update.txt file if it doesn't exist
if [ ! -f "${SCRIPT_DIR}/last_update.txt" ]; then
    date +%Y-%m-%d > "${SCRIPT_DIR}/last_update.txt"
    echo "Initialized last_update.txt with today's date"
fi