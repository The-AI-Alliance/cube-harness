# Install the Azure VM Agent (WaAppAgent) so that when Azure creates a VM from
# this image, the agent can:
#   * apply the os_profile admin_password at first boot
#   * report VM provisioning status back to ARM
#   * accept future VM extensions (CustomScriptExtension, etc.)
#
# Without this agent the launch() path will time out waiting for SSH even when
# the authorized_keys are baked in, because admin_password is not applied and
# Azure's provisioner never marks the VM as Ready.
#
# Idempotent: skips if WindowsAzureGuestAgent service is already registered.

$ErrorActionPreference = 'Stop'

if (Get-Service -Name WindowsAzureGuestAgent -ErrorAction SilentlyContinue) {
    Write-Host 'Azure VM Agent already installed - skipping.'
    exit 0
}

$agentUri = 'https://go.microsoft.com/fwlink/?LinkID=394789'
$msiPath  = 'C:\Windows\Temp\WindowsAzureVmAgent.msi'

# Prefer a cached MSI pre-staged by the Packer file provisioner from
# ~/.cube/cache/ on the host. Falls back to a guest-side download through
# QEMU's bandwidth-limited user-mode networking.
if (Test-Path $msiPath) {
    Write-Host "Using pre-staged $msiPath (skipping Microsoft download)."
} else {
    Write-Host 'Downloading Azure VM Agent...'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $agentUri -OutFile $msiPath -UseBasicParsing
}

Write-Host 'Installing Azure VM Agent (silent)...'
$proc = Start-Process -FilePath 'msiexec.exe' `
    -ArgumentList '/i', $msiPath, '/quiet', '/norestart' `
    -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    throw "msiexec failed with exit code $($proc.ExitCode)"
}

Remove-Item -Force $msiPath
Write-Host 'Azure VM Agent installed.'
