@echo off
:: Log start time
echo %DATE% %TIME% - Starting main bot execution >> logs\batch_execution.log

:: Set current directory to batch file location
cd /d %~dp0

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run the main bot execution script
python earnings_calendar_bot.py

:: Log completion
echo %DATE% %TIME% - Main bot execution completed >> logs\batch_execution.log