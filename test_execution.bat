@echo off
echo ===== EARNINGS BOT TEST EXECUTION =====
echo Current directory: %CD%
echo Current time: %TIME%
echo Current date: %DATE%

:: Create logs directory if it doesn't exist
if not exist logs mkdir logs

echo.
echo ===== ACTIVATING VIRTUAL ENVIRONMENT =====
call venv\Scripts\activate.bat
echo.
echo ===== PYTHON VERSION =====
python --version
echo.
echo ===== TESTING IBKR CONNECTION =====
python test_ibkr.py
echo.
echo ===== TESTING PRESCREENING WITH ONE SYMBOL =====
python -c "from earnings_calendar_bot import analyze_stock_metrics; print(analyze_stock_metrics('KHC'))"
echo.
echo ===== TEST COMPLETE =====
echo Test completed successfully at %TIME% >> logs\batch_execution.log
pause