#!/bin/sh
# Auto-add DCO Signed-off-by line if not already present.
# Skips merge, squash, and fixup commits.
COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="$2"
[ "$COMMIT_SOURCE" = "merge" ] && exit 0
[ "$COMMIT_SOURCE" = "squash" ] && exit 0

SOB="Signed-off-by: $(git config user.name) <$(git config user.email)>"
grep -qF "$SOB" "$COMMIT_MSG_FILE" || printf "\n%s\n" "$SOB" >> "$COMMIT_MSG_FILE"
