<#
.SYNOPSIS
  Windows equivalent of the Makefile.
.EXAMPLE
  .\run.ps1 install
  .\run.ps1 test
  .\run.ps1 test-int
  .\run.ps1 lint
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "test", "test-int", "lint", "format", "fixture", "ingest", "clean", "help")]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Fixture = "tests\fixtures\handbook.pdf"

function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    "install" {
        $base = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
        Invoke-Step $base @("-m", "venv", ".venv")
        Invoke-Step $Py @("-m", "pip", "install", "-e", ".[dev]")
    }
    "test" {
        Invoke-Step $Py @("-m", "pytest", "tests/unit", "-m", "not integration", "--cov",
            "--cov-report=term-missing", "--cov-fail-under=85")
    }
    "test-int" { Invoke-Step $Py @("-m", "pytest", "tests/integration", "-m", "integration") }
    "lint" {
        Invoke-Step $Py @("-m", "ruff", "check", ".")
        Invoke-Step $Py @("-m", "ruff", "format", "--check", ".")
    }
    "format" {
        Invoke-Step $Py @("-m", "ruff", "check", "--fix", ".")
        Invoke-Step $Py @("-m", "ruff", "format", ".")
    }
    "fixture" { Invoke-Step $Py @("scripts/make_fixture_pdf.py") }
    "ingest" { Invoke-Step $Py @("-m", "app.ingest", $Fixture) }
    "clean" {
        foreach ($p in @("data", ".pytest_cache", ".coverage", "htmlcov")) {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
    }
    default {
        Write-Host "usage: .\run.ps1 <install|test|test-int|lint|format|fixture|ingest|clean>"
    }
}
