# ./scripts/load_python_env.ps1

Write-Host ""
Write-Host "Restoring frontend npm packages"
Write-Host ""
Set-Location ./app/frontend
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to restore frontend npm packages"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Building frontend"
Write-Host ""
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build frontend"
    exit $LASTEXITCODE
}


Write-Host ""
Write-Host "Starting backend"
Write-Host ""
Set-Location ../backend
Set-Location ../.. # Change to project root
$venvPythonPath = "./.venv/Scripts/python.exe" # Updated Windows venv Python path
if (Test-Path -Path "/usr") {
  # fallback to Linux venv path
  $venvPythonPath = "./.venv/bin/python" # Updated Linux venv Python path
}
$backendHost = "127.0.0.1"
$backendPort = "8765"
Write-Host "Starting FastAPI backend on $backendHost`:$backendPort using Uvicorn..."
Start-Process -FilePath $venvPythonPath -ArgumentList "-m uvicorn app.backend.main:app --host $backendHost --port $backendPort --reload" -Wait -NoNewWindow
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start backend"
    exit $LASTEXITCODE
}
