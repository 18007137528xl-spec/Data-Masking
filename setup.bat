@echo off
REM Double-click entry point for Windows.
REM
REM A .ps1 file cannot be double-clicked: Windows opens it in an editor, and
REM even from a shell the default execution policy blocks it. This launcher
REM invokes PowerShell with the policy bypassed for this one process only --
REM it does not change any machine or user setting.
REM
REM It also translates the POSIX-style flags (--core) that setup.sh and the
REM README use into the PowerShell-style switches (-Core) that setup.ps1
REM declares. Both spellings work here, because a user who has read the
REM README should not have to know which shell is underneath.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "ARGS="
:parse
if "%~1"=="" goto run
set "A=%~1"
if /i "!A!"=="--core"        set "A=-Core"
if /i "!A!"=="--skip-model"  set "A=-SkipModel"
if /i "!A!"=="--skip-check"  set "A=-SkipSelfCheck"
if /i "!A!"=="--help"        set "A=-?"
if /i "!A!"=="-h"            set "A=-?"
set "ARGS=!ARGS! !A!"
shift
goto parse

:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"!ARGS!
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo Setup exited with code %RC%. Scroll up for the first red [FAIL] line.
    echo.
)

REM The pause exists so a double-clicked window does not vanish before the
REM result can be read. Under CI there is nobody to press a key, so skip it --
REM otherwise the job hangs until it times out.
if defined CI goto :done
echo Press any key to close this window.
pause >nul

:done
endlocal & exit /b %RC%
