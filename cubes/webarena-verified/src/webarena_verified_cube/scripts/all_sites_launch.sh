#!/usr/bin/env bash
# Launch all 6 WebArena-Verified sites on a single Docker host.
# All images are pre-pulled during provision; volumes are pre-populated via VolumeSpec.
set -euo pipefail

echo "[launch] Starting all 6 WebArena-Verified containers …"

dump_service_debug() {
    local name="$1"
    local container="webarena_${name}"
    echo "[launch][debug] $name container status:"
    docker ps -a --filter "name=${container}" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
    echo "[launch][debug] $name recent container logs:"
    docker logs --tail 200 "${container}" 2>&1 || true
    if [ "$name" = "wikipedia" ]; then
        echo "[launch][debug] wikipedia /data details:"
        docker exec webarena_wikipedia sh -lc \
            'ls -lah /data && stat /data/wikipedia_en_all_maxi_2022-05.zim || true' 2>&1 || true
        echo "[launch][debug] wikipedia kiwix version:"
        docker exec webarena_wikipedia sh -lc 'kiwix-serve --version || true' 2>&1 || true
    fi
}

# ── shopping_admin (Magento admin) ────────────────────────────────────────────
docker run -d --name webarena_shopping_admin \
    -p 7780:80 -p 7781:8877 \
    am1n3e/webarena-verified-shopping_admin

# ── shopping (Magento storefront) ─────────────────────────────────────────────
docker run -d --name webarena_shopping \
    -p 7770:80 -p 7771:8877 \
    am1n3e/webarena-verified-shopping

# ── reddit (Postmill) ─────────────────────────────────────────────────────────
docker run -d --name webarena_reddit \
    -p 9999:80 -p 9998:8877 \
    am1n3e/webarena-verified-reddit

# ── gitlab ────────────────────────────────────────────────────────────────────
docker run -d --name webarena_gitlab \
    -p 8023:8023 -p 8024:8877 \
    am1n3e/webarena-verified-gitlab

# ── wikipedia (Kiwix — volume pre-populated by VolumeSpec) ────────────────────
docker run -d --platform linux/arm64/v8 --name webarena_wikipedia \
    -p 8888:8080 -p 8889:8874 \
    -v webarena_wikipedia_data:/data \
    ghcr.io/kiwix/kiwix-serve:3.8.0 \
    /data/wikipedia_en_all_maxi_2022-05.zim

# ── map (OSM + Nominatim + OSRM — 9 volumes pre-populated by VolumeSpec) ─────
docker run -d --name webarena_map \
    -p 3000:8080 -p 3001:8877 \
    -v webarena_map_tile_db:/data/database \
    -v webarena_map_routing_car:/data/routing/car \
    -v webarena_map_routing_bike:/data/routing/bike \
    -v webarena_map_routing_foot:/data/routing/foot \
    -v webarena_map_nominatim_db:/data/nominatim/postgres \
    -v webarena_map_nominatim_flatnode:/data/nominatim/flatnode \
    -v webarena_map_website_db:/var/lib/postgresql/14/main \
    -v webarena_map_tiles:/data/tiles \
    -v webarena_map_style:/data/style \
    am1n3e/webarena-verified-map

echo "[launch] All containers started, waiting for healthchecks …"

# ── Healthcheck helper ────────────────────────────────────────────────────────
wait_healthy() {
    local name="$1" url="$2" max_attempts="$3"
    for i in $(seq 1 "$max_attempts"); do
        curl -sf "$url" > /dev/null 2>&1 && echo "[launch] $name healthy" && return 0
        sleep 2
    done
    echo "ERROR: $name did not become healthy after $((max_attempts * 2))s" >&2
    dump_service_debug "$name"
    exit 1
}

# Sites in parallel (backgrounded), then wait for all — capture PIDs to
# propagate individual failures (bare `wait` only returns the last job's exit code).
pids=()
wait_healthy "shopping_admin" "http://localhost:7780/"                     150 & pids+=($!)
wait_healthy "shopping"       "http://localhost:7770/customer/account/login" 150 & pids+=($!)
wait_healthy "reddit"         "http://localhost:9999/login"                 60  & pids+=($!)
wait_healthy "gitlab"         "http://localhost:8023/users/sign_in"         300 & pids+=($!)
wait_healthy "wikipedia"      "http://localhost:8888/"                      60  & pids+=($!)
wait_healthy "map"            "http://localhost:3000/"                      60  & pids+=($!)
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "ERROR: one or more sites failed healthcheck" >&2; exit 1; }

echo "[launch] All 6 sites healthy"

# Magento sites need extra warmup for PHP caches
sleep 30

# Relax Reddit rate limits so the agent doesn't get throttled.
# See https://github.com/gasse/webarena-setup/commit/b4426309
docker exec webarena_reddit sh -lc '
if [ -f /srv/forum/app/DataSource/SubmissionData.php ]; then
  sed -i \
    -e "s/1 hour/2 minutes/g" \
    -e "s/5 minutes/2 minutes/g" \
    -e "s/max=3/max=50/g" \
    -e "s/max=15/max=50/g" \
    /srv/forum/app/DataSource/SubmissionData.php
else
  echo "[launch][warn] Missing /srv/forum/app/DataSource/SubmissionData.php; skipping reddit rate-limit patch"
fi
' || true

docker exec webarena_reddit sh -lc '
if [ -f /srv/forum/app/DataSource/CommentData.php ]; then
  sed -i \
    -e "s/5 minutes/2 minutes/g" \
    -e "s/max=10/max=50/g" \
    /srv/forum/app/DataSource/CommentData.php
else
  echo "[launch][warn] Missing /srv/forum/app/DataSource/CommentData.php; skipping reddit rate-limit patch"
fi
' || true

docker exec webarena_reddit sh -lc '
if [ -f /srv/forum/app/DataSource/UserData.php ]; then
  sed -i \
    -e '"'"'s/max="3"/max="50"/g'"'"' \
    -e "s/1 hour/2 minutes/g" \
    /srv/forum/app/DataSource/UserData.php
else
  echo "[launch][warn] Missing /srv/forum/app/DataSource/UserData.php; skipping reddit rate-limit patch"
fi
' || true

docker exec webarena_reddit sh -lc '
if [ -f /srv/forum/bin/console ]; then
  php /srv/forum/bin/console cache:clear
else
  echo "[launch][warn] Missing /srv/forum/bin/console; skipping cache:clear"
fi
' || true
docker exec webarena_reddit php -r "opcache_reset();" 2>/dev/null || true

echo "[launch] Warmup complete — ready"
