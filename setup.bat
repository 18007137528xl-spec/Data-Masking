@echo off
REM Double-click entry point for Windows.
REM
REM A .ps1 file cannot be double-clicked: Windows opens it in an editor, and
REM even from a shell the default execution policy blocks it. This launcher
REM invokes PowerShell with the policy bypassed for this one process only --
REM it does not change any machine or user setting.

setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*

echo.
echo Press any key to close this window.
pause >nul
endlocal
