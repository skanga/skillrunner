param([switch]$RemoveShells)
$ErrorActionPreference = 'Stop'
if ($env:RUNNER_ENVIRONMENT -ne 'github-hosted' -or $env:RUNNER_OS -ne 'Windows') {
    throw 'This script is restricted to disposable GitHub-hosted Windows runners.'
}
$evidence = Join-Path $env:RUNNER_TEMP 'no-posix-evidence'
New-Item -ItemType Directory -Force $evidence | Out-Null
$names = @('bash.exe', 'sh.exe', 'zsh.exe', 'dash.exe', 'busybox.exe')
$drives = @(Get-CimInstance Win32_LogicalDisk | Where-Object DriveType -eq 3 | ForEach-Object { $_.DeviceID + '\' })
$phase = if ($RemoveShells) { 'before' } else { 'after' }
$inventoryPath = Join-Path $evidence "inventory-$phase.json"
& .venv/Scripts/python.exe scripts/inventory_posix_shells.py $inventoryPath @drives
if ($LASTEXITCODE) { throw 'Shell inventory failed' }
$inventory = Get-Content -Raw $inventoryPath | ConvertFrom-Json
$found = @($inventory.shells)
Write-Output "Inventory exclusions (retained for review): $($inventory.inaccessible.Count)"
$distributions = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' -ErrorAction SilentlyContinue | ForEach-Object { (Get-ItemProperty $_.PSPath).DistributionName } | Where-Object { $_ })
$distributions | ConvertTo-Json -AsArray | Set-Content (Join-Path $evidence 'wsl-distributions.json')
if ($distributions.Count) { throw "Unexpected WSL distributions: $($distributions -join ', ')" }
# Windows' signed WSL launchers are not POSIX shell implementations. They are
# allowed only with no registered distribution; never mutate Windows servicing files.
$launchers = @()
$shells = @()
foreach ($path in ($found | Sort-Object -Unique)) {
    $isWindowsLauncher = [IO.Path]::GetFileName($path) -eq 'bash.exe' -and (
        $path -eq "$env:SystemRoot\System32\bash.exe" -or
        $path -eq "$env:SystemRoot\SysWOW64\bash.exe" -or
        $path.StartsWith("$env:SystemRoot\WinSxS\", [StringComparison]::OrdinalIgnoreCase)
    )
    if ($isWindowsLauncher) {
        $signature = Get-AuthenticodeSignature -LiteralPath $path
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft') {
            throw "Unverified Windows launcher: $path"
        }
        $launchers += $path
    } else { $shells += $path }
}
$phase = if ($RemoveShells) { 'before' } else { 'after' }
@{phase=$phase; drives=$drives; shells=$shells; windows_wsl_launchers=$launchers; wsl_distributions=$distributions; commit=$env:GITHUB_SHA} |
    ConvertTo-Json -Depth 5 | Set-Content (Join-Path $evidence "$phase.json")
if ($RemoveShells) {
    if (-not $shells.Count) { throw 'Expected a stock hosted image containing POSIX shells for the removal check.' }
    foreach ($path in $shells) {
        Write-Output "Removing POSIX shell executable: $path"
        Remove-Item -LiteralPath $path -Force
        if (Test-Path -LiteralPath $path) { throw "Removal failed: $path" }
    }
} elseif ($shells.Count) {
    throw "POSIX shell executables remain: $($shells -join ', ')"
} else {
    foreach ($name in $names) {
        foreach ($command in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
            if ($command.Source -notin $launchers) { throw "Unexpected shell on PATH: $($command.Source)" }
        }
    }
    Write-Output 'Verified: no POSIX shell implementations found on fixed drives; no runner-account WSL distributions.'
}
