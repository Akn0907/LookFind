@echo off
cd /d "%~dp0"
if not exist .venv (
  echo [LookFind] Sanal ortam olusturuluyor...
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if not exist .env (
  copy .env.example .env >nul
  echo.
  echo [LookFind] .env dosyasi olusturuldu.
  echo SERPAPI_API_KEY ve ROBOFLOW_API_KEY degerlerini Not Defteri ile girin.
  echo Kaydedip start.bat dosyasini tekrar calistirin.
  pause
  exit /b
)
echo.
echo LookFind V0.4.2: http://127.0.0.1:8000
echo Kapatmak icin bu pencerede CTRL+C.
echo.
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
