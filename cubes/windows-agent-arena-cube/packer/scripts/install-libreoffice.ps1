# Install LibreOffice 24.8.2.1 from the official Document Foundation MSI.
#
# Why this script exists at all: the upstream WAA setup.ps1 attempts this
# install during data.img build, but (a) saves the download with a `.exe`
# extension and pipes it through `msiexec /i`, which silently no-ops, and
# (b) the upstream mirror URLs (stable/24.8.2/) have since been moved off
# the documentfoundation `stable/` path — so even with the .exe/.msi bug
# fixed, a fresh setup.ps1 run would 404. We layer LibreOffice on in Packer.
#
# Speed: the MSI is ~346 MB. Through QEMU's user-mode networking we measured
# ~8 KB/s of effective throughput (a guest-side download would take ~12 hrs).
# The Packer file provisioner pre-stages the MSI from the host's
# ~/.cube/cache/ to C:\Windows\Temp\ before this script runs; we use that
# cached file when present and only fall back to a guest download otherwise.
#
# Idempotent: skips if the soffice binary is already present.

$ErrorActionPreference = 'Stop'

$sofficePath = 'C:\Program Files\LibreOffice\program\soffice.exe'
if (Test-Path $sofficePath) {
    Write-Output "LibreOffice already installed at $sofficePath - skipping."
    exit 0
}

# 24.8.2 is no longer in /stable/; downloadarchive serves the .1 build.
# These are fallback URLs only — the file provisioner should normally
# pre-stage the MSI to $msiPath.
$mirrors = @(
    'https://downloadarchive.documentfoundation.org/libreoffice/old/24.8.2.1/win/x86_64/LibreOffice_24.8.2.1_Win_x86-64.msi'
)
$msiPath = 'C:\Windows\Temp\LibreOffice_24.8.2.1_Win_x86-64.msi'

if (Test-Path $msiPath) {
    Write-Output "Using pre-staged $msiPath (skipping download)."
} else {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $downloaded = $false
    foreach ($url in $mirrors) {
        Write-Output "Downloading LibreOffice from $url ..."
        try {
            Invoke-WebRequest -Uri $url -OutFile $msiPath -UseBasicParsing -TimeoutSec 600
            if ((Get-Item $msiPath).Length -gt 100MB) {
                $downloaded = $true
                break
            }
            Write-Output "Download from $url too small ($((Get-Item $msiPath).Length) bytes) - trying next mirror."
            Remove-Item -Force $msiPath -ErrorAction SilentlyContinue
        } catch {
            Write-Output "Download from $url failed: $_"
            Remove-Item -Force $msiPath -ErrorAction SilentlyContinue
        }
    }
    if (-not $downloaded) {
        throw "All LibreOffice mirrors failed."
    }
}

# Match upstream WAA setup.ps1 exactly: just `/i FILE /quiet`, no extra
# MSI properties. Anything else (CREATEDESKTOPLINK, REGISTER_ALL_MSO_TYPES,
# /norestart) would diverge the installed state from what entry.sh would
# have produced on a successful first install.
Write-Output 'Installing LibreOffice (silent, upstream-equivalent args)...'
$proc = Start-Process -FilePath 'msiexec.exe' `
    -ArgumentList '/i', "`"$msiPath`"", '/quiet' `
    -Wait -PassThru
# 3010 = "success, reboot requested". msiexec returns it on some LibreOffice
# subcomponent installs; treat as success since Packer will issue its own
# shutdown after provisioners and a deferred reboot is harmless.
if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
    throw "msiexec failed with exit code $($proc.ExitCode)"
}

Remove-Item -Force $msiPath

if (-not (Test-Path $sofficePath)) {
    throw "LibreOffice install reported success but $sofficePath is missing."
}

Write-Output 'Adding C:\Program Files\LibreOffice\program to system PATH...'
$libreBin = 'C:\Program Files\LibreOffice\program'
$systemPath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
if ($systemPath -notlike "*$libreBin*") {
    [Environment]::SetEnvironmentVariable('Path', "$systemPath;$libreBin", 'Machine')
}

Write-Output "LibreOffice installed at $sofficePath."
