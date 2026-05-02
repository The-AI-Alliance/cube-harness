# Install the build-time SSH public key at the shared administrators' authorized
# keys file. For admin-group users, Windows OpenSSH reads keys from
# C:\ProgramData\ssh\administrators_authorized_keys - NOT from ~/.ssh/.
#
# Strict ACLs are mandatory: sshd refuses to use the file unless owner and
# read access are limited to Administrators + SYSTEM with no inheritance.
#
# Packer's `file` provisioner uploads the source pubkey to C:\Windows\Temp
# before this script runs.

$ErrorActionPreference = 'Stop'

$sshDir    = 'C:\ProgramData\ssh'
$authKeys  = Join-Path $sshDir 'administrators_authorized_keys'
$uploaded  = 'C:\Windows\Temp\id_rsa.pub'

if (-not (Test-Path $uploaded)) {
    throw "Uploaded pubkey not found at $uploaded - the Packer file provisioner should have placed it."
}

New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
Copy-Item -Force $uploaded $authKeys

Write-Host "Applying required ACLs to $authKeys..."
icacls $authKeys /inheritance:r | Out-Null
icacls $authKeys /grant 'Administrators:F' 'SYSTEM:F' | Out-Null

Remove-Item -Force $uploaded
Write-Host "authorized_keys installed at $authKeys."
