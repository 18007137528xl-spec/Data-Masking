@echo off
REM Double-click to start the review console, then use it in the browser.
REM
REM Typing three commands into PowerShell is where this went wrong in practice:
REM a mistyped line, a key set in one window and the console started in
REM another, a folder that was not the repository. This does the same three
REM things, the same way every time.
REM
REM The vault key, in order:
REM   1. DEIDKIT_KEY_URI or DEIDKIT_VAULT_KEY if already set on this machine --
REM      how a production server should be configured, the key never on disk
REM      beside the vault;
REM   2. otherwise the development key setup wrote to out\dev-vault.key, with
REM      a warning, because that key sits beside the vault it opens;
REM   3. otherwise stop and say so.
REM
REM Extra arguments pass through, e.g.  serve.bat --port 9000

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\deidkit.exe" (
    echo deidkit is not installed in this folder yet.
    echo Double-click setup.bat first, then run this again.
    echo.
    pause
    exit /b 1
)

if defined DEIDKIT_KEY_URI goto start
if defined DEIDKIT_VAULT_KEY goto start

if exist "out\dev-vault.key" (
    set /p DEIDKIT_VAULT_KEY=<"out\dev-vault.key"
    echo Using the DEVELOPMENT vault key in out\dev-vault.key.
    echo   It is stored beside the vault it opens. That is fine for synthetic
    echo   data; for real data the key belongs in a key service, set on this
    echo   machine as DEIDKIT_KEY_URI. Ask whoever runs this server.
    echo.
    goto start
)

echo No vault key was found.
echo   Neither DEIDKIT_KEY_URI nor DEIDKIT_VAULT_KEY is set, and there is no
echo   out\dev-vault.key. Ask whoever runs this server how the key is provided.
echo.
pause
exit /b 1

:start
echo Starting the console. Your browser will open it in a moment.
echo Keep this window open while you work -- closing it stops the console.
echo.
".venv\Scripts\deidkit.exe" serve --open %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    pause
)
endlocal & exit /b %RC%
