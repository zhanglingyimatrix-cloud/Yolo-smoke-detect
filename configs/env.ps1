$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot "configs\ultralytics"
$env:MPLCONFIGDIR = Join-Path $ProjectRoot "configs\matplotlib"
$env:PYTHONUTF8 = "1"
