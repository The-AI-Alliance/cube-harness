# Docker Service Provisioning — Design Overview

## Problem

Benchmarks like WebArena-Verified need Docker containers running on remote VMs (AWS EC2, Azure VMs). Most sites (shopping_admin, reddit, gitlab, etc.) only need a Docker image — pull it, run it. But two sites have **large external data dependencies**:

- **Wikipedia**: a ~80 GB ZIM file that must be mounted into the container at `/data`
- **MAP** (OpenStreetMap + Nominatim + OSRM): 3 large archives from S3, extracted into 9 separate Docker volumes (tile database, routing data for car/bike/foot, Nominatim PostgreSQL + flatnode, and 3 empty runtime volumes)

### Why the previous API couldn't support MAP

Before `VolumeSpec`, `DockerServiceConfig` had exactly three fields relevant to provisioning and launching:

```python
docker_images: list[str]   # pulled during provision, baked into AMI
launch_script: str         # bash run at launch time via SSH
services: dict[str, int]   # ports to SSH-tunnel
```

The provisioning step only did one thing: **pull Docker images**. There was no mechanism to download archives, create Docker volumes, or extract data during provisioning.

The only alternative was to put the data setup into `launch_script` — download the tarballs and populate volumes every time a VM launches. For MAP, that means downloading tens of GB from S3 and extracting into 9 volumes on every single run. This is:

1. **Slow** — adds 30-60+ minutes to every launch, turning a 3-minute boot into a 30+ minute wait
2. **Wasteful** — re-downloads the same static data every time, burning network bandwidth and S3 egress
3. **Fragile** — a network hiccup during a large download kills the launch with no resume
4. **Expensive** — larger instance types needed to hold the data, with longer runtime per launch

The right fix is to do the data setup **once during provisioning** and bake it into the VM snapshot. That's what `VolumeSpec` enables — it tells the infra backend "download this archive, extract this subpath into this Docker volume" as part of the provision step. The result is baked into the AMI/gallery image. Every subsequent launch boots from that snapshot with all data already present.

## Core Concepts

### Resource / Infra Separation

```
ResourceConfig  — WHAT the benchmark needs (benchmark-owned, declarative)
InfraConfig     — HOW to provision and launch it (infra-owned, executable)
ResourceHandle  — Live runtime handle (returned by launch())
```

The benchmark author declares resources. The infra backend provisions and launches them. This separation means the same `DockerServiceConfig` works on AWS, Azure, or any future cloud — the benchmark doesn't know or care which.

### DockerServiceConfig

A `DockerServiceConfig` represents a Docker service stack on a remote VM:

```python
DockerServiceConfig(
    name="webarena-all",
    docker_images=["img1", "img2", ...],   # pre-pulled during provision
    services={"web": 7780, "ctrl": 7781},  # SSH-tunneled to localhost
    volumes=[VolumeSpec(...)],             # data downloaded during provision
    launch_script="docker run -d ...",     # runs at launch time
    endpoint_to_site={"web": "site_name"}, # benchmark-specific routing
)
```

### VolumeSpec — Provision-Time Data

`VolumeSpec` declares a named Docker volume that should be created and optionally populated from a remote archive during provisioning:

```python
VolumeSpec(
    name="webarena_map_tile_db",
    mount_path="/data/database",
    source_url="https://s3.amazonaws.com/osm_tile_server.tar",
    tar_subpath="projects/ogma3/docker/volumes/osm-data/_data",
    strip_components=6,
)
```

- `source_url`: archive to download (skipped if already present — idempotent)
- `tar_subpath` + `strip_components`: which part of the archive to extract
- Archives referenced by multiple VolumeSpecs are downloaded once
- Volumes without `source_url` are created empty (populated at container runtime)

### Lifecycle

```
provision()                    — runs ONCE, result baked into VM snapshot
  1. Launch bootstrap VM
  2. Install Docker
  3. docker pull all images
  4. For each VolumeSpec: download archive, create volume, extract data
  5. Snapshot VM → AMI / Gallery Image
  6. Register in ProvisionStore

launch()                       — runs EVERY time
  1. Boot VM from snapshot (images + volumes already present)
  2. Run launch_script via SSH (starts containers)
  3. Open SSH tunnels for each service port
  4. Return ResourceHandle with endpoints

handle.close()                 — tears down the VM
unprovision()                  — deletes the snapshot (manual, intentional)
```

## WebArena-Verified: Concrete Example

`WEBARENA_ALL` bundles all 6 sites into one `DockerServiceConfig`:

| Component | Count |
|-----------|-------|
| Docker images | 6 (shopping_admin, shopping, reddit, gitlab, wikipedia, map) |
| Service ports | 12 (web UI + env-ctrl per site) |
| VolumeSpecs | 10 (1 Wikipedia ZIM + 9 MAP volumes from 3 archives) |

One provision cycle creates a VM snapshot with everything pre-loaded. Each subsequent launch boots from that snapshot, starts all 6 containers, and tunnels all 12 ports.

## Future-Proofing for Other Cubes

### What this design handles well

- **Single-container benchmarks** (e.g., SWE-bench, MLE-bench): One image, no volumes, simple launch script. `DockerServiceConfig` works out of the box.
- **Multi-container stacks** (e.g., TheAgentCompany): Multiple images and services in one config, one launch script that starts them all. Same pattern as WebArena.
- **Data-heavy benchmarks**: `VolumeSpec` handles arbitrary archives. Any benchmark that needs pre-loaded data (datasets, model weights, database dumps) can declare it.
- **Cloud-agnostic**: Same `DockerServiceConfig` works on AWS and Azure (and any future infra backend). The benchmark author doesn't write cloud-specific code.

### What would require extension

- **GPU containers**: `DockerImageConfig` (the per-task variant) already has a `gpu` field. `DockerServiceConfig` could add `--gpus all` to docker run if needed.
- **Docker Compose**: If a benchmark needs complex container orchestration (networks, depends_on, healthcheck dependencies), the current model requires encoding it in `launch_script`. A future `compose_content` field could handle this more cleanly.
- **Volume snapshots across runs**: Currently volumes are baked into the AMI. If a benchmark needs to reset volumes between runs (e.g., fresh database per experiment), the current model requires re-provisioning. A future enhancement could support COW volume overlays.

## Brittleness Risks

### 1. launch_script is opaque bash
The infra backend can't validate or inspect `launch_script` — it's a black box. Bugs in the script (wrong ports, missing volume mounts) only surface at runtime. **Mitigation**: integration tests that exercise the full lifecycle.

### 2. Volume archive URLs are hardcoded
If upstream archives move or change format, provisioning breaks. **Mitigation**: `source_url` is in the benchmark's `resources.py`, not deep in infra code — easy to find and update.

### 3. Large AMIs
Baking all data into the AMI means large snapshots (especially MAP). This increases storage costs and snapshot creation time. **Mitigation**: per-site resources (`WEBARENA_SHOPPING_ADMIN`, etc.) allow provisioning only what you need for a subset of tasks.

### 4. SSH tunnel fragility
All service access goes through SSH tunnels. If a tunnel drops mid-experiment, the benchmark task fails. **Mitigation**: `ServerAliveInterval=30` keeps tunnels alive; the infra backend logs tunnel state.

### 5. Single-VM model
All containers run on one VM. If a container is memory-hungry (GitLab, MAP), it competes with others. **Mitigation**: use a large instance type. For production runs, the benchmark could be configured with per-site resources instead of `WEBARENA_ALL`.

## File Map

```
cube-standard/
  src/cube/resource.py          — VolumeSpec, DockerServiceConfig definitions
  src/cube/infra_utils.py       — build_volume_setup_script() helper
  cube-resources/
    cube-infra-aws/             — AWSInfraConfig (provision/launch for EC2)
    cube-infra-azure/           — AzureInfraConfig (provision/launch for Azure VMs)

cube-harness/
  cubes/webarena-verified/
    src/webarena_verified_cube/
      resources.py              — WEBARENA_ALL + per-site resource declarations
      scripts/                  — launch scripts (bash)
      benchmark.py              — infra= auto-provisioning in _setup()
      debug.py                  — get_debug_benchmark(infra=) for testing
  integration-tests/
    test_webarena_debug_azure.py  — end-to-end Azure test
    test_webarena_debug_aws.py    — end-to-end AWS test
```
