# Quick start for Windows: install uv if needed, create a project in the current
# directory, add soundscapy and JupyterLab, then run the first-plot script.
# SOUNDSCAPY_SPEC overrides the requirement (used by CI before the first release).
$ErrorActionPreference = "Stop"

$Spec = if ($env:SOUNDSCAPY_SPEC) { $env:SOUNDSCAPY_SPEC } else { "soundscapy==1.0.0.dev1" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv init --bare
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv add $Spec jupyterlab
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run python _quickstart.py
exit $LASTEXITCODE
