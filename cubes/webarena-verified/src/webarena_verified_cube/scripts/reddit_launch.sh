#!/usr/bin/env bash
# Launch script for webarena-verified reddit container.
set -euo pipefail

docker run -d \
    --name webarena_reddit \
    -p 9999:80 \
    -p 9998:8877 \
    am1n3e/webarena-verified-reddit

healthy=0
for i in $(seq 1 60); do
    curl -sf http://localhost:9999/login > /dev/null 2>&1 && echo "healthy" && healthy=1 && break
    sleep 2
done
if [ "$healthy" -eq 0 ]; then
    echo "ERROR: reddit did not become healthy after 120s" >&2
    exit 1
fi

# Relax Reddit rate limits so the agent doesn't get throttled.
# See https://github.com/gasse/webarena-setup/commit/b4426309
docker exec webarena_reddit sed -i \
    -e "s/1 hour/2 minutes/g" \
    -e "s/5 minutes/2 minutes/g" \
    -e "s/max=3/max=50/g" \
    -e "s/max=15/max=50/g" \
    /srv/forum/app/DataSource/SubmissionData.php

docker exec webarena_reddit sed -i \
    -e "s/5 minutes/2 minutes/g" \
    -e "s/max=10/max=50/g" \
    /srv/forum/app/DataSource/CommentData.php

docker exec webarena_reddit sed -i \
    -e 's/max="3"/max="50"/g' \
    -e "s/1 hour/2 minutes/g" \
    /srv/forum/app/DataSource/UserData.php

docker exec webarena_reddit php /srv/forum/bin/console cache:clear
docker exec webarena_reddit php -r "opcache_reset();" 2>/dev/null || true
