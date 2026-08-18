@echo off
echo ====================================================================
echo VeritasAI - Public Sharable Link Generator
echo ====================================================================
echo.
echo Select public tunneling option to share this web app publicly:
echo.
echo [1] Localtunnel (No signup needed, runs via npx)
echo [2] Cloudflare Tunnel (Fast and secure)
echo [3] Ngrok (Requires ngrok installed)
echo.
set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" (
    echo.
    echo Starting Localtunnel on port 8000...
    echo Your public link will be generated below:
    npx localtunnel --port 8000
) else if "%choice%"=="2" (
    echo.
    echo Starting Cloudflare Tunnel on http://localhost:8000...
    cloudflared tunnel --url http://localhost:8000
) else if "%choice%"=="3" (
    echo.
    echo Starting Ngrok on port 8000...
    ngrok http 8000
) else (
    echo Invalid choice.
)

pause
