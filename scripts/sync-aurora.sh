#!/usr/bin/env bash
# Push the working tree to aurora for a test run.
#
# Iteration transport only -- git stays for milestones. Code is a few hundred KB, so
# this takes well under a second over Tailscale, whereas commit/push/pull per edit
# makes the history unreadable and the feedback loop slow.
#
# Data, checkpoints and virtualenvs are excluded and are NOT deleted on the remote by
# --delete: rsync leaves excluded paths alone. aurora owns its own data/ and .venv.
#
#   scripts/sync-aurora.sh            # sync
#   scripts/sync-aurora.sh -n         # dry run, show what would change
set -euo pipefail

HOST="${AURORA_HOST:-adam@aurora}"
DEST="${AURORA_PATH:-work/distrain}"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -az --delete "$@" \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'data/' \
    --exclude 'out/' \
    --exclude 'checkpoints/' \
    --exclude 'trackio/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude 'reference/_clones/' \
    "$LOCAL/" "$HOST:$DEST/"

echo "synced $LOCAL -> $HOST:$DEST"
