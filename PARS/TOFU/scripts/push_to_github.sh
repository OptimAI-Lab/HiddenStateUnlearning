#!/usr/bin/env bash
set -euo pipefail

# push_to_github.sh
# Creates a git repo locally (if missing), commits, creates remote (via gh if available),
# and pushes to GitHub. Defaults to user/repo: jiajunruan/Probe_unlearning.
# Usage examples:
#   GITHUB_USER=jiajunruan REPO_NAME=Probe_unlearning bash scripts/push_to_github.sh
#   GITHUB_TOKEN=ghp_xxx GITHUB_USER=jiajunruan REPO_NAME=Probe_unlearning bash scripts/push_to_github.sh

USER=${GITHUB_USER:-jiajunruan}
REPO=${REPO_NAME:-Probe_unlearning}
REMOTE="https://github.com/${USER}/${REPO}.git"

# Check git
command -v git >/dev/null 2>&1 || { echo "git not found; please install git."; exit 1; }

# Initialize repo if not already
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Initializing git repository..."
  git init
fi

# Stage and commit
git add .
if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "Initial commit: Probe_unlearning code" || true
fi

# Use GitHub CLI if available to create repo and push
if command -v gh >/dev/null 2>&1; then
  echo "Found gh CLI — attempting to create repo and push via gh..."
  # will prompt or use gh auth if configured
  gh repo create ${USER}/${REPO} --public --source=. --remote=origin --push || echo "gh create/push failed or repo exists; continuing..."
fi

# If origin not present, add origin (with token if provided)
if ! git remote | grep -q origin; then
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    AUTH_REMOTE="https://${GITHUB_TOKEN}@github.com/${USER}/${REPO}.git"
    git remote add origin "$AUTH_REMOTE"
  else
    git remote add origin "$REMOTE"
  fi
fi

# Ensure main branch
git branch -M main || true

# Push
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git push -u origin main --force
else
  git push -u origin main
fi

echo "Pushed to https://github.com/${USER}/${REPO}"
