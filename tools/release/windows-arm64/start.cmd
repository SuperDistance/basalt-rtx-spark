@echo off
rem Starts the server of a basalt-rtx-spark release the way upstream Basalt runs a release archive:
rem bin\basalt.exe serve --config basalt_config.json, with the .venv beside bin\ as the server's Python and the CUDA
rem libraries in bin\. Arguments go to the server, for example "start.cmd --port 8081". Stop it with Ctrl+C.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The server's Python environment is missing: run setup.cmd first.
  goto fail
)
rem every pack basalt_config.json names must exist before the engine starts loading
".venv\Scripts\python.exe" -c "import json,os,sys; c=json.load(open('basalt_config.json',encoding='utf-8')); m=[x['model'] for x in c.get('models',[]) if not os.path.isfile(x.get('model',''))]; [print('Model pack not found:',x) for x in m]; sys.exit(1 if m else 0)"
if errorlevel 1 (
  echo Run setup.cmd to download it, or set "model" in basalt_config.json to the pack you have.
  goto fail
)
echo Starting the server. When it prints "weights loaded", open its "stats" address in a browser, or point an
echo OpenAI or Anthropic client at its "ready" address with the model "flash-next".
"bin\basalt.exe" serve --config basalt_config.json %*
if errorlevel 1 goto fail
exit /b 0

:fail
rem a double-clicked window closes on exit: keep it open so the message can be read
echo %cmdcmdline% | "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul && pause
exit /b 1
