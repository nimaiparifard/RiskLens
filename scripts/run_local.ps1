<#
Run the RiskLens stack natively on Windows (no Docker): MLflow UI, serving, Django, Vite.
Prerequisites: .venv with all packages (uv sync), trained + promoted models in mlflow.db
(or models/*/scorer.joblib as fallback), frontend\node_modules.

  powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1          # start
  powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -Stop    # stop

URLs: app http://localhost:5173 · API docs http://localhost:8000/api/docs · MLflow http://localhost:5000
SQLite replaces Postgres, Celery tasks are run on demand (management commands / API).
#>
param([switch]$Stop, [switch]$Seed)

$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Logs = Join-Path $Root ".cache\local"
New-Item -ItemType Directory -Force $Logs | Out-Null
$PidFile = Join-Path $Logs "pids.txt"

if ($Stop) {
    if (Test-Path $PidFile) {
        Get-Content $PidFile | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
        Remove-Item $PidFile
    }
    Write-Host "stopped"; return
}

$db = ($Root -replace '\\', '/') + "/mlflow.db"
$env:MLFLOW_TRACKING_URI = "sqlite:///$db"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:PYTHONIOENCODING = "utf-8"
$token = "local-dev-token"
$pids = @()

# MLflow UI / REST (the Django /api/models endpoint reads the registry through it)
$pids += (Start-Process $Py -ArgumentList "-m", "mlflow", "server", "--backend-store-uri", "`"sqlite:///$db`"",
    "--host", "127.0.0.1", "--port", "5000" -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$Logs\mlflow.out" -RedirectStandardError "$Logs\mlflow.log").Id

# Serving: champion from the registry, local scorer files as fallback
$env:MODEL_SOURCE = "mlflow"; $env:INTERNAL_TOKEN = $token; $env:LOG_JSON = "false"
$env:CREDIT_MODEL_PATH = "$Root\models\credit\scorer.joblib"; $env:FRAUD_MODEL_PATH = "$Root\models\fraud\scorer.joblib"
$pids += (Start-Process $Py -ArgumentList "-m", "uvicorn", "app.main:create_app", "--factory", "--port", "8001" `
    -WorkingDirectory "$Root\serving" -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$Logs\serving.out" -RedirectStandardError "$Logs\serving.log").Id

# Django (SQLite)
$env:DJANGO_SETTINGS_MODULE = "config.settings.dev"
$env:DATABASE_URL = "sqlite:///$($Root -replace '\\','/')/.cache/local/risklens.sqlite3"
$env:SERVING_URL = "http://127.0.0.1:8001"; $env:SERVING_INTERNAL_TOKEN = $token
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
& $Py "$Root\backend\manage.py" migrate --noinput | Out-Null
$pids += (Start-Process $Py -ArgumentList "manage.py", "runserver", "127.0.0.1:8000", "--noreload" `
    -WorkingDirectory "$Root\backend" -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$Logs\backend.out" -RedirectStandardError "$Logs\backend.log").Id

# Vite dev server (proxies /api to :8000)
$nw = Join-Path $Root ".venv\Lib\site-packages\nodejs_wheel"
$node = if (Get-Command node -ErrorAction SilentlyContinue) { "node" } else { "$nw\node.exe" }
$env:PATH = "$nw;$env:PATH"
$pids += (Start-Process $node -ArgumentList "node_modules\vite\bin\vite.js", "--host", "127.0.0.1", "--port", "5173", "--strictPort" `
    -WorkingDirectory "$Root\frontend" -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$Logs\vite.out" -RedirectStandardError "$Logs\vite.log").Id

$pids | Set-Content $PidFile

foreach ($url in "http://127.0.0.1:8001/health/ready", "http://127.0.0.1:8000/api/health/", "http://127.0.0.1:5173") {
    for ($i = 0; $i -lt 90; $i++) {
        try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; Write-Host "ready: $url"; break } catch { Start-Sleep 1 }
    }
}
if ($Seed) { & $Py "$Root\backend\manage.py" seed_demo --applications 60 --score }
Write-Host "app: http://localhost:5173  (users admin / officer / viewer, password: risklens-demo)"
