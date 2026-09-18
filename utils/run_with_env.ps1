param(
    [ValidatePattern('^[A-Za-z0-9_]+$')]
    [string]$Strategy = 'LAA_MA4',
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'

try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'Project Python is missing. Run setup_env.bat first.'
    }

    # Existing process settings take precedence, then Windows User and Machine.
    foreach ($name in @('FRED_API_KEY', 'UNRATE_VINTAGES_CSV')) {
        $value = [Environment]::GetEnvironmentVariable($name, 'Process')
        if ([string]::IsNullOrWhiteSpace($value)) {
            foreach ($scope in @('User', 'Machine')) {
                $value = [Environment]::GetEnvironmentVariable($name, $scope)
                if (-not [string]::IsNullOrWhiteSpace($value)) { break }
            }
        }
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            [Environment]::SetEnvironmentVariable($name, $value.Trim(), 'Process')
        }
    }

    if ($env:UNRATE_VINTAGES_CSV) {
        Set-Location -LiteralPath $projectRoot
        if (-not (Test-Path -LiteralPath $env:UNRATE_VINTAGES_CSV -PathType Leaf)) {
            throw 'UNRATE_VINTAGES_CSV is set but the file does not exist.'
        }
        Write-Host 'Using UNRATE vintage CSV.'
    } else {
        if (-not $env:FRED_API_KEY) {
            if ($Check) { throw 'FRED_API_KEY was not found in Process, User or Machine settings.' }
            $secret = Read-Host 'Enter FRED API key (hidden; used only for this run)' -AsSecureString
            $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
            try {
                $env:FRED_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim()
            } finally {
                [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
                $secret.Dispose()
            }
        }
        if ($env:FRED_API_KEY -cnotmatch '^[a-z0-9]{32}$') {
            throw 'FRED_API_KEY must be 32 lowercase letters/digits, without quotes.'
        }
        Write-Host 'FRED_API_KEY loaded (value hidden).'
    }

    Set-Location -LiteralPath $projectRoot
    if ($Check) {
        # Confirm that Python actually inherits the setting; no API request.
        & $pythonPath -c "import os, sys; ready = bool(os.environ.get('FRED_API_KEY') or os.environ.get('UNRATE_VINTAGES_CSV')); print('Python configuration available:', ready); sys.exit(0 if ready else 1)"
    } else {
        Write-Host "Running backtest: $Strategy"
        & $pythonPath (Join-Path $projectRoot 'runBacktest.py') $Strategy
    }
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
