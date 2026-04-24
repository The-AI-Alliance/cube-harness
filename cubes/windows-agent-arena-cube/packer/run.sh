#!/usr/bin/env bash
# Wrapper that sets up per-build pflash vars + swtpm, runs packer, and cleans up.
#
# Why: the qemu Packer builder doesn't know how to manage (a) a writable copy of
# OVMF_VARS per build or (b) a swtpm daemon alongside the VM. Both are required
# for a Windows 11 guest (UEFI + TPM 2.0). This script handles the external
# state; Packer drives the actual VM lifecycle.
#
# Usage:
#   export PKR_VAR_admin_password='…'
#   ./run.sh
#
# Env vars honored:
#   SOURCE_QCOW2   — path to base image (default ~/.cube/images/waa-windows-vm.qcow2)
#   SSH_PUBKEY     — public key to bake into authorized_keys (default ~/.ssh/id_rsa.pub)
#   PKR_VAR_admin_password — REQUIRED, password for Docker user on base image

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

: "${PKR_VAR_admin_password:?Set PKR_VAR_admin_password to the base image Docker-user password}"

SOURCE_QCOW2="${SOURCE_QCOW2:-$HOME/.cube/images/waa-windows-vm.qcow2}"
# Default SSH key: prefer id_ed25519, fall back to id_rsa. Override via SSH_PUBKEY.
if [[ -z "${SSH_PUBKEY:-}" ]]; then
    for candidate in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub"; do
        [[ -r "$candidate" ]] && { SSH_PUBKEY="$candidate"; break; }
    done
fi

[[ -r "$SOURCE_QCOW2" ]] || { echo "source qcow2 not readable: $SOURCE_QCOW2" >&2; exit 1; }
[[ -n "${SSH_PUBKEY:-}" && -r "$SSH_PUBKEY" ]] || { echo "no SSH pubkey found — generate one with ssh-keygen, or set SSH_PUBKEY env var" >&2; exit 1; }

# Packer extracts plugin binaries into $TMPDIR and then execs them. Ubuntu's
# default /tmp is mounted noexec, which breaks plugin loading. Use a user-local
# dir unless the caller overrode TMPDIR already.
if [[ -z "${TMPDIR:-}" ]] || findmnt -n -o OPTIONS --target "$TMPDIR" 2>/dev/null | grep -q noexec; then
    export TMPDIR="$HOME/.packer-tmp"
    mkdir -p "$TMPDIR"
fi
command -v swtpm  >/dev/null || { echo "swtpm not installed (apt install swtpm swtpm-tools)" >&2; exit 1; }
command -v packer >/dev/null || { echo "packer not installed (see README.md)"                >&2; exit 1; }

workdir="$(mktemp -d /tmp/waa-packer-XXXXXX)"
pflash_vars="$workdir/OVMF_VARS.fd"
tpm_dir="$workdir/tpm"
tpm_sock="$tpm_dir/sock"

mkdir -p "$tpm_dir"
cp /usr/share/OVMF/OVMF_VARS_4M.ms.fd "$pflash_vars"

cleanup() {
    local code=$?
    if [[ -n "${swtpm_pid:-}" ]] && kill -0 "$swtpm_pid" 2>/dev/null; then
        kill "$swtpm_pid" 2>/dev/null || true
        wait "$swtpm_pid" 2>/dev/null || true
    fi
    rm -rf "$workdir"
    exit "$code"
}
trap cleanup EXIT INT TERM

echo "[run.sh] Starting swtpm on $tpm_sock..."
swtpm socket \
    --tpmstate "dir=$tpm_dir" \
    --ctrl "type=unixio,path=$tpm_sock" \
    --tpm2 \
    --log "file=$tpm_dir/swtpm.log,level=20" &
swtpm_pid=$!

# Wait for the socket to appear (up to 5s).
for _ in $(seq 1 50); do
    [[ -S "$tpm_sock" ]] && break
    sleep 0.1
done
[[ -S "$tpm_sock" ]] || { echo "swtpm socket never appeared at $tpm_sock" >&2; exit 1; }

echo "[run.sh] swtpm ready (pid=$swtpm_pid)"

packer init .
packer build \
    -var "source_qcow2=$SOURCE_QCOW2" \
    -var "ssh_pubkey_path=$SSH_PUBKEY" \
    -var "pflash_vars_path=$pflash_vars" \
    -var "tpm_socket_path=$tpm_sock" \
    waa-windows.pkr.hcl
