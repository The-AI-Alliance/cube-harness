# windows-agent-arena-cube — Design Decisions

## Q&A from Planning Session

**Q: The WAA evaluation code (getters/metrics/SetupController) lives in `desktop_env/` inside the WAA repo and uses a `vm_ip` string interface. Should we port those files into the cube package (modifying them to use GuestAgent), or import them directly from the WAA repo?**

> **A: Port into cube** — Copy getters/metrics/SetupController into waa-cube and adapt them to use GuestAgent. Self-contained, follows the osworld-cube pattern exactly. No runtime dependency on WAA repo structure.

---

**Q: The WAA tasks use named QEMU snapshots (e.g. `'vscode'`, `'libreoffice_calc'`). Should we implement the `WAADockerVMBackend` (which uses QMP `loadvm` for snapshot restoration) inside the waa-cube package, or as a new backend in the shared `cube-vm-backend` package?**

> **A: Inside waa-cube** — Self-contained, no changes to cube-vm-backend. The WAA-specific QMP snapshot logic stays co-located with the WAA benchmark code.

---

**Q: Is the `windowsarena/winarena:latest` Docker image already available (built locally or pullable from Docker Hub)?**

> **A: Needs to be pulled/built** — `ensure_resource()` handles validating Docker image availability and raises a clear error with build instructions if the image is not present.
