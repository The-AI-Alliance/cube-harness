# Side-load OpenSSH Server from the official GitHub release ZIP, register the
# service, open firewall, set default shell.
#
# Why not Add-WindowsCapability? The WAA base image has Windows Update disabled
# (upstream setup explicitly breaks WU to stop automatic updates mid-eval), so
# Add-WindowsCapability -Online hangs forever trying to contact wuauserv. The
# side-load path pulls the same binaries directly from GitHub and avoids the
# Windows Servicing Stack entirely.

$ErrorActionPreference = 'Stop'

$release = 'https://github.com/PowerShell/Win32-OpenSSH/releases/download/v9.5.0.0p1-Beta/OpenSSH-Win64.zip'
$zipPath = 'C:\Windows\Temp\OpenSSH-Win64.zip'
$dstDir  = 'C:\Program Files\OpenSSH'

if (Test-Path (Join-Path $dstDir 'sshd.exe')) {
    Write-Output "OpenSSH already installed at $dstDir - skipping download."
} else {
    # Prefer a cached zip pre-staged by the Packer file provisioner from
    # ~/.cube/cache/ on the host. Falls back to a guest-side GitHub download —
    # which is bandwidth-limited by QEMU's user-mode networking (~8 KB/s).
    if (Test-Path $zipPath) {
        Write-Output "Using pre-staged $zipPath (skipping GitHub download)."
    } else {
        Write-Output "Downloading OpenSSH Server from GitHub..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $release -OutFile $zipPath -UseBasicParsing
    }

    Write-Output "Extracting to $dstDir..."
    if (Test-Path $dstDir) { Remove-Item -Recurse -Force $dstDir }
    Expand-Archive -Path $zipPath -DestinationPath 'C:\Program Files\' -Force
    # Archive extracts as OpenSSH-Win64; rename.
    if (Test-Path 'C:\Program Files\OpenSSH-Win64') {
        Move-Item -Force 'C:\Program Files\OpenSSH-Win64' $dstDir
    }
    Remove-Item -Force $zipPath
}

Write-Output 'Registering sshd + ssh-agent services via install-sshd.ps1...'
& "$dstDir\install-sshd.ps1"

Write-Output 'Configuring services (auto-start)...'
Set-Service -Name sshd -StartupType Automatic
Set-Service -Name ssh-agent -StartupType Automatic
Start-Service sshd
Start-Service ssh-agent

Write-Output 'Ensuring firewall rule for port 22...'
if (-not (Get-NetFirewallRule -Name sshd -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
}

Write-Output 'Setting default shell to PowerShell for SSH sessions...'
$openSshKey = 'HKLM:\SOFTWARE\OpenSSH'
if (-not (Test-Path $openSshKey)) {
    New-Item -Path $openSshKey -Force | Out-Null
}
New-ItemProperty -Path $openSshKey -Name 'DefaultShell' `
    -Value 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -PropertyType String -Force | Out-Null

Write-Output 'OpenSSH Server installed and running (side-loaded build).'
