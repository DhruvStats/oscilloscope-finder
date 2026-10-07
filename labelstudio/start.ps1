# Start Label Studio for this project (local only: http://localhost:8080).
# Database and settings live in labelstudio/data (git-ignored); photos are served from raw/ without copying.
$root = Split-Path -Parent $PSScriptRoot
$env:LABEL_STUDIO_BASE_DATA_DIR = Join-Path $root "labelstudio\data"
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = Join-Path $root "raw"
# run through python -m so the environment keeps working if the project folder is moved
& (Join-Path $root ".labelstudio-venv\Scripts\python.exe") -m label_studio.server start --port 8080 --internal-host localhost
