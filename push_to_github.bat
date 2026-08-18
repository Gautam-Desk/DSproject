@echo off
echo ===================================================
echo   Pushing VeritasAI Project to GitHub Repository
echo   Remote: https://github.com/Gautam-Desk/DSproject.git
echo ===================================================
echo.

git remote remove origin 2>nul
git remote add origin https://github.com/Gautam-Desk/DSproject.git
git branch -M main
git push -u origin main

echo.
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Repository pushed to https://github.com/Gautam-Desk/DSproject.git successfully!
) else (
    echo [NOTE] If prompted, sign in via your browser or paste your GitHub Personal Access Token (PAT).
)
echo.
pause
