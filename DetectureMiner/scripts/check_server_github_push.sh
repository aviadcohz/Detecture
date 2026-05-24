#!/usr/bin/env bash
set -euo pipefail

# Verify GitHub connectivity and push permissions for the current repo.

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Run this script inside a git repository." >&2
  exit 1
fi

ORIGIN_URL="$(git remote get-url origin)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

log "repo: $(pwd)"
log "branch: ${BRANCH}"
log "origin: ${ORIGIN_URL}"

echo
log "Checking read access: git ls-remote origin"
git ls-remote --heads origin >/dev/null
log "Read access OK"

echo
log "Checking push permissions (dry-run): git push --dry-run origin ${BRANCH}"
git push --dry-run origin "${BRANCH}" >/dev/null
log "Push dry-run OK"

echo
log "GitHub push setup is healthy for this repo."
