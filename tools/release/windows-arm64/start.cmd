@echo off
rem Starts the server of a basalt-rtx-spark release with basalt_config.json (the CUDA libraries are in bin\ beside the
rem engine). Stop it with Ctrl+C.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.cmd first.
  exit /b 1
)
if not exist "models\Qwen3.8-FN-UD-IQ3_XXS.basalt" (
  echo The model pack is not in models\ yet: run setup.cmd, or point basalt_config.json at your pack.
)
"bin\basalt.exe" serve --config basalt_config.json %*
