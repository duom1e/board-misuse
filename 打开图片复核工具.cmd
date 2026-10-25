@echo off
cd /d "%~dp0"
python review_app.py
if errorlevel 1 pause
