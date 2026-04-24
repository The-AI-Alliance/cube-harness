# Make the image self-contained: copy the WAA Flask guest-agent code into
# C:\waa\setup\ and re-register the WindowsArena_OnLogon scheduled task to run
# it from there instead of the SMB share \\host.lan\Data\on-logon.ps1.
#
# Why: the upstream WAA image reads its guest-agent code from an SMB share the
# upstream WAA Docker container serves. Our LocalInfraConfig path and the Azure
# launch path both boot the VM without any SMB share, so the on-logon scheduled
# task fails silently and port 5000 never opens. Once this provisioner runs, the
# image boots to a fully-functional agent under plain QEMU or on Azure.
#
# Packer's `file` provisioner uploads the WAA upstream `vm/setup/` tree to
# C:\Windows\Temp\waa-setup\ before this script runs.

$ErrorActionPreference = 'Stop'

$srcRoot  = 'C:\Windows\Temp\waa-setup'
$dstRoot  = 'C:\waa\setup'
$taskName = 'WindowsArena_OnLogon'

if (-not (Test-Path $srcRoot)) {
    throw "WAA setup tree not found at $srcRoot - the Packer `file` provisioner should have uploaded it."
}

Write-Output "--- copying WAA setup tree: $srcRoot -> $dstRoot ---"
if (Test-Path $dstRoot) {
    Remove-Item -Recurse -Force $dstRoot
}
New-Item -ItemType Directory -Force -Path $dstRoot | Out-Null
Copy-Item -Recurse -Force "$srcRoot\*" $dstRoot

# Rewrite on-logon.ps1 to point at the local path. The upstream script hardcodes
# `$scriptFolder = "\\host.lan\Data"` at line 1; we replace it with $dstRoot.
$onLogonPath = Join-Path $dstRoot 'on-logon.ps1'
Write-Output "--- patching $onLogonPath to use local paths ---"
$content = Get-Content $onLogonPath -Raw
$patched = $content -replace '(?m)^\s*\$scriptFolder\s*=\s*"[^"]+"', "`$scriptFolder = `"$dstRoot`""
Set-Content -Path $onLogonPath -Value $patched -Encoding UTF8

# Re-register the scheduled task to use the local on-logon.ps1. If the task
# already exists (from a prior run of install.bat via SMB) it'll point at
# \\host.lan\Data - unregister it first so our version wins.
Write-Output "--- re-registering scheduled task $taskName ---"
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-ExecutionPolicy Bypass -File `"$onLogonPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User 'Docker'
$principal = New-ScheduledTaskPrincipal -UserId 'Docker' -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName `
    -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null

# Allow TCP inbound on port 5000 (install.bat via SMB normally creates this
# firewall rule during first-run; if the image never went through that path on
# our build host, it may be missing).
$fwName = 'PythonHTTPServer-5000'
if (-not (Get-NetFirewallRule -Name $fwName -ErrorAction SilentlyContinue)) {
    Write-Output "--- adding firewall rule $fwName ---"
    New-NetFirewallRule -Name $fwName -DisplayName $fwName `
        -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Any | Out-Null
}

Remove-Item -Recurse -Force $srcRoot
Write-Output 'waa-server embedded; on-logon task rewired to local paths'
