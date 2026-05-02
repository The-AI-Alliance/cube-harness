# Configure AutoAdminLogon for the Docker admin user after the bootstrap step
# changed the Docker password.
#
# Windows looks at TWO places for the auto-login password on boot:
#   1. HKLM:\...\Winlogon\DefaultPassword         (plaintext)
#   2. The LSA "DefaultPassword" secret           (encrypted via LsaStorePrivateData)
#
# When both exist the LSA secret wins. The upstream WAA unattend stored the
# original (empty) Docker password into the LSA secret. After bootstrap_winrm
# changed the Docker password, that LSA secret is stale - so Windows tries the
# old empty password, fails silently, and no one is signed in at boot. The
# scheduled task WindowsArena_OnLogon (AtLogOn trigger) never fires and the
# WAA Flask guest agent on :5000 never starts.
#
# This script overwrites the LSA secret with the new plaintext password AND
# sets the registry values for completeness.

$ErrorActionPreference = 'Stop'

$adminUser = $env:ADMIN_USER
$adminPassword = $env:ADMIN_PASSWORD
if (-not $adminUser -or -not $adminPassword) {
    throw 'ADMIN_USER and ADMIN_PASSWORD env vars must be set.'
}

$winlogon = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'

Write-Output "Setting Winlogon registry (AutoAdminLogon=1, DefaultUserName=$adminUser)"
New-ItemProperty -Path $winlogon -Name DefaultUserName -Value $adminUser -PropertyType String -Force | Out-Null
New-ItemProperty -Path $winlogon -Name DefaultPassword -Value $adminPassword -PropertyType String -Force | Out-Null
New-ItemProperty -Path $winlogon -Name AutoAdminLogon -Value '1' -PropertyType String -Force | Out-Null
New-ItemProperty -Path $winlogon -Name DefaultDomainName -Value '' -PropertyType String -Force | Out-Null
New-ItemProperty -Path $winlogon -Name AutoLogonCount -Value 9999 -PropertyType DWord -Force | Out-Null

Write-Output 'Storing LSA secret "DefaultPassword"...'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class Lsa
{
    [StructLayout(LayoutKind.Sequential)]
    public struct LSA_OBJECT_ATTRIBUTES
    {
        public int Length;
        public IntPtr RootDirectory;
        public IntPtr ObjectName;
        public int Attributes;
        public IntPtr SecurityDescriptor;
        public IntPtr SecurityQualityOfService;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LSA_UNICODE_STRING
    {
        public ushort Length;
        public ushort MaximumLength;
        [MarshalAs(UnmanagedType.LPWStr)]
        public string Buffer;
    }

    [DllImport("advapi32.dll", SetLastError=true)]
    public static extern uint LsaOpenPolicy(
        IntPtr SystemName,
        ref LSA_OBJECT_ATTRIBUTES ObjectAttributes,
        uint DesiredAccess,
        out IntPtr PolicyHandle);

    [DllImport("advapi32.dll", SetLastError=true)]
    public static extern uint LsaStorePrivateData(
        IntPtr PolicyHandle,
        ref LSA_UNICODE_STRING KeyName,
        ref LSA_UNICODE_STRING PrivateData);

    [DllImport("advapi32.dll", SetLastError=true)]
    public static extern uint LsaClose(IntPtr ObjectHandle);

    // POLICY_CREATE_SECRET = 0x00000020
    public const uint POLICY_CREATE_SECRET = 0x20;

    public static void Store(string key, string value)
    {
        LSA_OBJECT_ATTRIBUTES attrs = new LSA_OBJECT_ATTRIBUTES();
        attrs.Length = Marshal.SizeOf(attrs);
        IntPtr handle;
        uint r = LsaOpenPolicy(IntPtr.Zero, ref attrs, POLICY_CREATE_SECRET, out handle);
        if (r != 0) throw new System.ComponentModel.Win32Exception((int)r, "LsaOpenPolicy failed ntstatus=0x" + r.ToString("X"));

        LSA_UNICODE_STRING k = new LSA_UNICODE_STRING();
        k.Buffer = key;
        k.Length = (ushort)(key.Length * 2);
        k.MaximumLength = (ushort)((key.Length + 1) * 2);

        LSA_UNICODE_STRING v = new LSA_UNICODE_STRING();
        v.Buffer = value;
        v.Length = (ushort)(value.Length * 2);
        v.MaximumLength = (ushort)((value.Length + 1) * 2);

        try {
            uint sr = LsaStorePrivateData(handle, ref k, ref v);
            if (sr != 0) throw new System.ComponentModel.Win32Exception((int)sr, "LsaStorePrivateData failed ntstatus=0x" + sr.ToString("X"));
        } finally {
            LsaClose(handle);
        }
    }
}
'@

[Lsa]::Store('DefaultPassword', $adminPassword)
Write-Output "AutoLogon configured for user $adminUser (registry + LSA secret both set)"
