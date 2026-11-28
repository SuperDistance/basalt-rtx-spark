@echo off
rem One-time setup of a basalt-rtx-spark release on Windows on Arm64: checks the Visual C++ runtime and Python, makes the
rem server's .venv beside bin\, and offers to download the model pack into models\.  Run it from the unzipped folder.
setlocal
cd /d "%~dp0"

if not exist "%SystemRoot%\System32\msvcp140.dll" (
  echo The Microsoft Visual C++ runtime for ARM64 is missing. Install it, then run setup.cmd again:
  echo   https://aka.ms/vs/17/release/vc_redist.arm64.exe
  goto fail
)

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Python 3.10 or newer is needed: https://www.python.org/downloads/windows/ ^(the ARM64 installer^)
  goto fail
)

if not exist ".venv\Scripts\python.exe" (
  echo Making the server's Python environment in .venv ...
  %PY% -m venv .venv || goto fail
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt huggingface_hub || goto fail
echo Python environment ready.

if exist "models\Qwen3.8-FN-UD-IQ3_XXS.basalt" goto done
echo.
set /p GET="Download the model pack now (Qwen3.8-FN-UD-IQ3_XXS, about 82 GiB)? [y/N] "
if /i not "%GET%"=="y" goto done
".venv\Scripts\hf.exe" download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-UD-IQ3_XXS.basalt --local-dir models || goto fail

:done
echo.
echo Setup done. Start the server with start.cmd, then open http://127.0.0.1:8080/
echo (the API is at http://127.0.0.1:8080/v1, model "flash-next").
call :hold
exit /b 0

:fail
call :hold
exit /b 1

rem a double-clicked window closes on exit: keep it open so the message can be read
:hold
echo %cmdcmdline% | "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul && pause
exit /b 0
