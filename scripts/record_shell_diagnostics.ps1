param([Parameter(Mandatory=$true)][string]$Phase)
$ErrorActionPreference = 'Stop'
if ($env:RUNNER_ENVIRONMENT -ne 'github-hosted' -or $env:RUNNER_OS -ne 'Windows') {
    throw 'This script is restricted to disposable GitHub-hosted Windows runners.'
}
$evidence = Join-Path $env:RUNNER_TEMP 'no-posix-evidence'
New-Item -ItemType Directory -Force $evidence | Out-Null
$paths = @()
foreach ($name in @('bash.exe', 'sh.exe', 'zsh.exe', 'dash.exe', 'busybox.exe')) {
    $paths += @(Get-Command $name -All -ErrorAction SilentlyContinue | ForEach-Object Source)
}
$paths += @(Get-ChildItem -LiteralPath "$env:LOCALAPPDATA\Microsoft\WindowsApps" -Filter bash.exe -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object FullName)
$records = @()
foreach ($path in ($paths | Sort-Object -Unique)) {
    $record = @{path=$path}
    try {
        $item = Get-Item -LiteralPath $path -Force
        $record.attributes = [string]$item.Attributes
        $record.length = $item.Length
        $record.link_type = $item.LinkType
        $record.target = $item.Target
        $record.created_utc = $item.CreationTimeUtc.ToString('o')
    } catch { $record.metadata_error = $_.Exception.Message }
    try {
        $signature = Get-AuthenticodeSignature -LiteralPath $path
        $record.signature_status = [string]$signature.Status
        $record.signer = if ($signature.SignerCertificate) { $signature.SignerCertificate.Subject } else { $null }
    } catch { $record.signature_error = $_.Exception.Message }
    try {
        $record.reparse_query = @(& "$env:SystemRoot\System32\fsutil.exe" reparsepoint query $path 2>&1 | ForEach-Object { "$_" })
        $record.reparse_query_exit = $LASTEXITCODE
    } catch { $record.reparse_error = $_.Exception.Message }
    $records += $record
}
$packages = @()
$packageError = $null
try {
    $packages = @(Get-AppxPackage -Name '*WindowsSubsystemForLinux*' | Select-Object Name, PackageFullName, PackageFamilyName, InstallLocation, SignatureKind, Status)
} catch { $packageError = $_.Exception.Message }
@{phase=$Phase; timestamp_utc=[DateTime]::UtcNow.ToString('o'); commit=$env:GITHUB_SHA; paths=$records; wsl_packages=$packages; package_error=$packageError} |
    ConvertTo-Json -Depth 8 | Set-Content (Join-Path $evidence "shell-diagnostics-$Phase.json")
# Diagnostic observations do not classify or exempt executables and never execute them.
exit 0
