#!/usr/bin/env bash
# Launch script for webarena-verified shopping (Magento storefront) container.
set -euo pipefail

docker run -d \
    --name webarena_shopping \
    -p 7770:80 \
    -p 7771:8877 \
    am1n3e/webarena-verified-shopping

healthy=0
for i in $(seq 1 150); do
    curl -sf http://localhost:7770/customer/account/login > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: shopping did not become healthy after 300s" >&2
    exit 1
fi
sleep 30
