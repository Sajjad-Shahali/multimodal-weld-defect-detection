$ErrorActionPreference = 'Stop'

# Always run dashboard from the repo venv
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

# Activate venv
& "$repo\weld\Scripts\Activate.ps1"

# Make weldml importable (src layout)
$env:PYTHONPATH = "$repo\dashboard\weld_project_template\src;$repo"

# Use ABSOLUTE path to config so it resolves regardless of CWD
$configPath = "$repo\dashboard\weld_project_template\configs\default.yaml"

# Run Streamlit
Set-Location "$repo\dashboard\weld_project_template"
streamlit run "src\weldml\dashboard\app.py" --server.headless true -- --config $configPath
