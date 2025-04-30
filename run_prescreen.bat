@echo off
:: Log start time
echo %DATE% %TIME% - Starting pre-screening >> logs\batch_execution.log

:: Set current directory to batch file location
cd /d %~dp0

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run the pre-screening script
python earnings_calendar_bot.py prescreen

:: Log completion
echo %DATE% %TIME% - Pre-screening completed >> logs\batch_execution.log