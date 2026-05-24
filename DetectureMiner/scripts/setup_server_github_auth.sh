#!/usr/bin/env bash
set -euo pipefail

# Configure GitHub push access on a new server (SSH-first).
# Safe defaults: does not overwrite existing keys and does not require interactive prompts.

KEY_PATH="${HOME}/.ssh/id_ed25519_detecture"
GIT_NAME=""
GIT_EMAIL=""
PRIVATE_REPO_SSH="git@github.com:scientific-computing-user/Detecture.git"
PUBLIC_REPO_SSH="git@github.com:scientific-computing-user/Detecture-site.git"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --git-name <name>          Set global git user.name
  --git-email <email>        Set global git user.email
  --key-path <path>          SSH key path (default: ${KEY_PATH})
  --private-repo-ssh <url>   Private repo SSH URL
  --public-repo-ssh <url>    Public site repo SSH URL
  -h, --help                 Show help

What this script does:
1) Creates SSH key (if missing) for GitHub pushes.
2) Prints public key so you can add it in GitHub -> Settings -> SSH keys.
3) Configures git identity if provided.
4) Converts current origin remote to SSH if it points to GitHub HTTPS.
5) Prints next validation command.
EOF
}

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --git-name)
      GIT_NAME="$2"
      shift 2
      ;;
    --git-email)
      GIT_EMAIL="$2"
      shift 2
      ;;
    --key-path)
      KEY_PATH="$2"
      shift 2
      ;;
    --private-repo-ssh)
      PRIVATE_REPO_SSH="$2"
      shift 2
      ;;
    --public-repo-ssh)
      PUBLIC_REPO_SSH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "$KEY_PATH")"
chmod 700 "$(dirname "$KEY_PATH")"

if [[ -n "$GIT_NAME" ]]; then
  git config --global user.name "$GIT_NAME"
  log "Configured git user.name"
fi
if [[ -n "$GIT_EMAIL" ]]; then
  git config --global user.email "$GIT_EMAIL"
  log "Configured git user.email"
fi

if [[ ! -f "$KEY_PATH" ]]; then
  COMMENT="${GIT_EMAIL:-detecture-server-key}"
  ssh-keygen -t ed25519 -C "$COMMENT" -f "$KEY_PATH" -N "" >/dev/null
  log "Created SSH key: $KEY_PATH"
else
  log "SSH key already exists: $KEY_PATH"
fi
chmod 600 "$KEY_PATH"
chmod 644 "${KEY_PATH}.pub"

# Ensure GitHub host key exists to avoid first-use prompt in automation.
ssh-keyscan -H github.com >> "${HOME}/.ssh/known_hosts" 2>/dev/null || true
chmod 644 "${HOME}/.ssh/known_hosts"

# If run inside the private repo, convert origin to SSH.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
  if [[ -n "$ORIGIN_URL" && "$ORIGIN_URL" == https://github.com/* ]]; then
    if [[ "$ORIGIN_URL" == *"Detecture-site"* ]]; then
      git remote set-url origin "$PUBLIC_REPO_SSH"
      log "Converted origin remote to SSH (public site repo)"
    else
      git remote set-url origin "$PRIVATE_REPO_SSH"
      log "Converted origin remote to SSH (private repo)"
    fi
  fi
fi

echo
log "Add this SSH public key to GitHub account (Settings -> SSH and GPG keys):"
echo
cat "${KEY_PATH}.pub"
echo

cat <<EOF
Next step:
1) Add key above in GitHub.
2) Validate push access:
   bash scripts/check_server_github_push.sh

If this repo is private code repo, expected SSH remote:
  $PRIVATE_REPO_SSH
If this is public site repo, expected SSH remote:
  $PUBLIC_REPO_SSH
EOF
