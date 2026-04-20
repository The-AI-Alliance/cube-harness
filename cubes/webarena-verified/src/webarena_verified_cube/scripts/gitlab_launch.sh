#!/usr/bin/env bash
# Launch script for webarena-verified GitLab container.
# NOTE: GitLab uses port 8023 internally (non-standard).
set -euo pipefail

docker run -d \
    --name webarena_gitlab \
    -p 8023:8023 \
    -p 8024:8877 \
    am1n3e/webarena-verified-gitlab

healthy=0
for i in $(seq 1 300); do
    curl -sf http://localhost:8023/users/sign_in > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: gitlab did not become healthy after 600s" >&2
    exit 1
fi
