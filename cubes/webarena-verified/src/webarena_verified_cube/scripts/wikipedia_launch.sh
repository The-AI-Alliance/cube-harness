#!/usr/bin/env bash
# Launch script for webarena-verified Wikipedia (Kiwix) container.
# Volumes are pre-populated by VolumeSpec during provision().
set -euo pipefail

docker run -d \
    --name webarena_wikipedia \
    -p 8888:8080 \
    -p 8889:8874 \
    -v webarena_wikipedia_data:/data \
    am1n3e/webarena-verified-wikipedia

healthy=0
for i in $(seq 1 60); do
    curl -sf http://localhost:8888/ > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: wikipedia did not become healthy after 120s" >&2
    exit 1
fi
