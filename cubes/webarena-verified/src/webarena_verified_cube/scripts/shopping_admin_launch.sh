#!/usr/bin/env bash
# Launch script for webarena-verified shopping_admin (Magento) container.
# Runs on the remote Docker host after provisioning.
set -euo pipefail

docker run -d \
    --name webarena_shopping_admin \
    -p 7780:80 \
    -p 7781:8877 \
    am1n3e/webarena-verified-shopping_admin

# Magento can take 3-5 min to initialise (up to 300s).
healthy=0
for i in $(seq 1 150); do
    curl -sf http://localhost:7780/ > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: shopping_admin did not become healthy after 300s" >&2
    exit 1
fi

# Extra warmup: Magento PHP caches take ~30s after first curl success before
# serving full pages to Playwright.
sleep 30
