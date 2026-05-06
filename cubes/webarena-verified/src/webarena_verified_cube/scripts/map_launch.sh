#!/usr/bin/env bash
# Launch script for webarena-verified MAP (OpenStreetMap + Nominatim + OSRM) container.
# All 9 volumes are pre-populated by VolumeSpec during provision().
set -euo pipefail

docker run -d \
    --name webarena_map \
    -p 3000:8080 \
    -p 3001:8877 \
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

healthy=0
for i in $(seq 1 60); do
    curl -sf http://localhost:3000/ > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: map did not become healthy after 120s" >&2
    exit 1
fi
