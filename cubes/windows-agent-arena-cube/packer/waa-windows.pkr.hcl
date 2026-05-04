packer {
  required_plugins {
    qemu = {
      source  = "github.com/hashicorp/qemu"
      version = "~> 1.0"
    }
  }
}

# ── Inputs ────────────────────────────────────────────────────────────────────

variable "source_qcow2" {
  type        = string
  description = "Base WAA Windows qcow2. Expected to have WinRM already enabled for admin_user. See README.md for one-time WinRM bootstrap."
  default     = "~/.cube/images/waa-windows-vm.qcow2"
}

variable "output_directory" {
  type    = string
  default = "output-waa-prepared"
}

variable "output_name" {
  type    = string
  default = "waa-windows-prepared"
}

variable "ssh_pubkey_path" {
  type        = string
  description = "Public key baked into C:\\ProgramData\\ssh\\administrators_authorized_keys at build time. Infra code at launch does NOT inject per-VM keys for Windows; the matching private key must live at ssh_privkey_path on the caller."
  default     = "~/.ssh/id_rsa.pub"
}

variable "admin_user" {
  type        = string
  description = "Existing admin user on the base image. Packer connects via WinRM as this user."
  default     = "Docker"
}

variable "admin_password" {
  type        = string
  sensitive   = true
  description = "Password for admin_user on the base image. Set via PKR_VAR_admin_password or -var."
}

variable "pflash_vars_path" {
  type        = string
  description = "Writable OVMF vars file (per-build copy). Created by run.sh wrapper."
}

variable "tpm_socket_path" {
  type        = string
  description = "UNIX socket of a running swtpm daemon. Created by run.sh wrapper."
}

variable "ovmf_code" {
  type    = string
  default = "/usr/share/OVMF/OVMF_CODE_4M.ms.fd"
}

# ── Source ────────────────────────────────────────────────────────────────────

# Boot the existing WAA qcow2 via a backing-file overlay so the source is never
# modified. UEFI + vTPM args mirror LocalInfraConfig.launch() so the guest boots
# identically to runtime.
source "qemu" "waa" {
  iso_url          = pathexpand(var.source_qcow2)
  iso_checksum     = "none"
  disk_image       = true
  use_backing_file = true
  format           = "qcow2"

  output_directory = var.output_directory
  vm_name          = "${var.output_name}.qcow2"

  accelerator  = "kvm"
  machine_type = "q35,smm=on"
  cpus         = 8
  memory       = 8192
  headless     = true
  display      = "none"

  # NOTE: Packer's qemu plugin "overlays" qemuargs over its defaults by first
  # token. Because we declare -drive entries here (for pflash), its auto-added
  # -drive for the main OS disk is dropped too — so we MUST also declare the
  # main disk explicitly, pointing at the overlay qcow2 Packer creates at
  # <output_directory>/<vm_name>.
  qemuargs = [
    ["-cpu", "host"],
    # Main OS disk — the backing-file overlay Packer writes to.
    ["-drive", "file=${var.output_directory}/${var.output_name}.qcow2,format=qcow2,if=virtio,cache=writeback,discard=ignore"],
    # UEFI firmware: read-only code + writable per-build vars (created by run.sh).
    ["-drive", "if=pflash,format=raw,readonly=on,file=${var.ovmf_code}"],
    ["-drive", "if=pflash,format=raw,file=${var.pflash_vars_path}"],
    # vTPM via swtpm (daemon + socket started by run.sh).
    ["-chardev", "socket,id=chrtpm,path=${var.tpm_socket_path}"],
    ["-tpmdev", "emulator,id=tpm0,chardev=chrtpm"],
    ["-device", "tpm-tis,tpmdev=tpm0"],
  ]

  # WinRM using admin credentials pre-baked in the source image.
  communicator   = "winrm"
  winrm_username = var.admin_user
  winrm_password = var.admin_password
  winrm_timeout  = "30m"

  # sysprep.ps1 uses /quit (not /shutdown) so the OS is still running when that
  # provisioner returns. Packer then issues its own explicit shutdown. 20m is
  # generous — a post-sysprep shutdown on Windows 11 is normally under a minute.
  shutdown_command = "shutdown /s /f /t 10 /c \"Packer shutdown\""
  shutdown_timeout = "20m"
}

# ── Build ─────────────────────────────────────────────────────────────────────

build {
  sources = ["source.qemu.waa"]

  # Upload build-time SSH pubkey; drop-authorized-keys.ps1 consumes it.
  provisioner "file" {
    source      = pathexpand(var.ssh_pubkey_path)
    destination = "C:/Windows/Temp/id_rsa.pub"
  }

  # Pre-stage installer artifacts from the host's ~/.cube/cache/ instead of
  # letting the per-script `Invoke-WebRequest` calls go through QEMU's
  # user-mode networking. WSMan upload is also slow (~5-13 KB/s measured)
  # but it's the only way to get artifacts in. LibreOffice is intentionally
  # excluded — it's already installed in data.img (Stage 2 setup.ps1).
  provisioner "file" {
    sources = [
      pathexpand("~/.cube/cache/OpenSSH-Win64.zip"),
      pathexpand("~/.cube/cache/WindowsAzureVmAgent.msi"),
    ]
    destination = "C:/Windows/Temp/"
  }

  # elevated_user/password: WinRM hands out a non-elevated (UAC-split) token by
  # default even for admin-group users, which blocks Add-WindowsCapability,
  # msiexec, sysprep, etc. Packer works around this by running each script via
  # a one-shot scheduled task under admin_user, which gets a full admin token.
  provisioner "powershell" {
    elevated_user     = var.admin_user
    elevated_password = var.admin_password
    environment_vars = [
      "ADMIN_USER=${var.admin_user}",
      "ADMIN_PASSWORD=${var.admin_password}",
    ]
    scripts = [
      "${path.root}/scripts/install-openssh-server.ps1",
      "${path.root}/scripts/install-azure-vm-agent.ps1",
      "${path.root}/scripts/drop-authorized-keys.ps1",
      "${path.root}/scripts/configure-autologon.ps1",
    ]
  }

  # NOTE: no sysprep. This image is designed to be used as a "Specialized"
  # Azure image (VMResourceConfig.specialized=True) — a byte-for-byte clone
  # preserving the Docker user, AutoLogon credentials, per-user app installs
  # (VS Code etc.), and every bit of profile state. Generalizing would wipe
  # exactly the state the agent sandbox needs, and Win11's strict AppX
  # sysprep checks refuse to generalize an image with per-user VS Code
  # installed anyway. scripts/sysprep.ps1 is kept in the repo for reference
  # but is not invoked.
}
