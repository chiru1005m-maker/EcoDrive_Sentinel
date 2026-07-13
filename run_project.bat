@echo off
echo Starting EcoDrive-Sentinel...
echo =========================================

:: Set the python path so the src module is found
set PYTHONPATH=.

:: Start the LangGraph Background Agent (Continuous Monitoring)
start "EcoDrive LangGraph Agent" .\venv_312\Scripts\python.exe src\agents\Sentinel_LangGraph.py --poll-ms 1000

:: Start the FastAPI Backend API in a new window
start "EcoDrive Backend API" .\venv_312\Scripts\python.exe src\api.py

:: Start the React Dashboard Server in a new window
start "EcoDrive Dashboard" .\venv_312\Scripts\python.exe src\dashboard\server.py

echo.
echo [ SUCCESS ]
echo The LangGraph Agent is running in the background.
echo The Backend API is running at:   http://localhost:8000/docs
echo The Dashboard is running at:     http://localhost:8081
echo.
echo To shut down the servers, just close the three new command prompt windows that popped up!
echo =========================================
pause
