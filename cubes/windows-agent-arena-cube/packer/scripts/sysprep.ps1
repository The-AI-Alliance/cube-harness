# Generalize the image. Sysprep removes per-install identifiers (machine SID,
# computer name, cached credentials) so Azure can apply a fresh os_profile at
# first deployment.
#
# /mode:vm is VM-optimised sysprep (no hardware driver cleanup, faster).
# /quit leaves the OS running after generalize so Packer's shutdown_command
# can drive the power-off explicitly — /shutdown inside sysprep creates a
# race between sysprep's internal work and Packer's shutdown wait, which
# routinely times out on Windows 11 images.

$ErrorActionPreference = 'Stop'

$adminUser = $env:ADMIN_USER
$adminPassword = $env:ADMIN_PASSWORD
if (-not $adminUser -or -not $adminPassword) {
    throw 'ADMIN_USER and ADMIN_PASSWORD env vars must be set (passed by Packer environment_vars).'
}

Write-Output "Clearing transient Azure VM Agent state (generalize hook)..."
$agentStateDir = 'C:\WindowsAzure'
if (Test-Path $agentStateDir) {
    Get-ChildItem $agentStateDir -Recurse -File -Include '*.xml', '*.json', '*.log' `
        -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

# Drop an unattend.xml that skips OOBE AND re-establishes Docker-user auto-login
# after generalize. Sysprep clears the LSA-encrypted DefaultPassword, so without
# <AutoLogon> the VM boots to a login screen with no one signed in — the
# WindowsArena_OnLogon scheduled task never fires and the guest agent never
# listens on :5000. Azure VM Agent sets its own os_profile credentials on top
# of this at first deploy, so this AutoLogon only matters for local QEMU boots
# (LocalInfraConfig smoke tests, etc.).
#
# Password is XML-escaped defensively in case the caller ever picks a value
# containing &, <, >, or quotes.
$pwXml = [System.Security.SecurityElement]::Escape($adminPassword)
$userXml = [System.Security.SecurityElement]::Escape($adminUser)

$unattend = @"
<?xml version=`"1.0`" encoding=`"utf-8`"?>
<unattend xmlns=`"urn:schemas-microsoft-com:unattend`">
  <settings pass=`"oobeSystem`">
    <component name=`"Microsoft-Windows-Shell-Setup`"
               processorArchitecture=`"amd64`"
               publicKeyToken=`"31bf3856ad364e35`"
               language=`"neutral`" versionScope=`"nonSxS`"
               xmlns:wcm=`"http://schemas.microsoft.com/WMIConfig/2002/State`"
               xmlns:xsi=`"http://www.w3.org/2001/XMLSchema-instance`">
      <OOBE>
        <HideEULAPage>true</HideEULAPage>
        <HideLocalAccountScreen>true</HideLocalAccountScreen>
        <HideOEMRegistrationScreen>true</HideOEMRegistrationScreen>
        <HideOnlineAccountScreens>true</HideOnlineAccountScreens>
        <HideWirelessSetupInOOBE>true</HideWirelessSetupInOOBE>
        <ProtectYourPC>3</ProtectYourPC>
      </OOBE>
      <AutoLogon>
        <Username>$userXml</Username>
        <Enabled>true</Enabled>
        <LogonCount>999</LogonCount>
        <Password>
          <Value>$pwXml</Value>
          <PlainText>true</PlainText>
        </Password>
      </AutoLogon>
      <TimeZone>UTC</TimeZone>
    </component>
  </settings>
</unattend>
"@

$unattendPath = "$env:windir\System32\Sysprep\unattend.xml"
$unattend | Out-File -FilePath $unattendPath -Encoding utf8 -Force

Write-Output 'Running sysprep (Packer will drive the subsequent shutdown)...'
& "$env:windir\System32\Sysprep\sysprep.exe" `
    /generalize /oobe /quit /mode:vm /quiet /unattend:$unattendPath

# sysprep.exe returns quickly but background work continues — wait for the
# process to be fully gone before we let Packer trigger shutdown.
Write-Output 'Waiting for sysprep to finish generalizing...'
$deadline = (Get-Date).AddMinutes(15)
while ((Get-Date) -lt $deadline) {
    $running = Get-Process sysprep -ErrorAction SilentlyContinue
    if (-not $running) { break }
    Start-Sleep -Seconds 10
}
Write-Output 'Sysprep completed.'
