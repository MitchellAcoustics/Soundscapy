# Quick start for Windows: install uv if needed, create a project in the current
# directory, add soundscapy and JupyterLab, then run _quickstart.py.
# SOUNDSCAPY_SPEC overrides the requirement (used by CI before the first release).
# Run with: powershell -ExecutionPolicy Bypass -File _quickstart.ps1
$ErrorActionPreference = "Stop"

$Spec = if ($env:SOUNDSCAPY_SPEC) { $env:SOUNDSCAPY_SPEC } else { "soundscapy==1.0.0.dev1" }

if (-not (Test-Path -PathType Leaf "_quickstart.py")) {
    throw "_quickstart.py not found: save the code from the install page as _quickstart.py in this folder, then run this again."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Test-Path -PathType Leaf "pyproject.toml")) {
    uv init --bare
    if ($LASTEXITCODE -ne 0) { throw "uv init failed (exit code $LASTEXITCODE)." }
}
uv add "$Spec" jupyterlab
if ($LASTEXITCODE -ne 0) { throw "uv add failed (exit code $LASTEXITCODE)." }
uv run python _quickstart.py
if ($LASTEXITCODE -ne 0) { throw "_quickstart.py failed (exit code $LASTEXITCODE)." }
