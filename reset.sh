#!/usr/bin/env bash
# Reset Luna to a clean, fresh state.
# Removes containers, images, venv, models, caches, and install metadata.
#
# Usage:
#   ./reset.sh              # interactive: asks before each step
#   ./reset.sh --yes        # non-interactive: removes everything
#   ./reset.sh --dry-run    # preview what would be deleted (safe)
#   ./reset.sh --dry-run --all  # preview everything including models/config
#
# NOT removed (your data):
#   config.json, memory/, history/, reminders/, knowledge/, agent/
#   To wipe everything including data: ./reset.sh --all
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
YES=0
ALL=0
FORCE=0
DRY=0

for arg in "$@"; do
    case "$arg" in
        --yes|-y)  YES=1 ;;
        --all)     ALL=1 ;;
        --force|-f) FORCE=1 ;;
        --dry-run|-n) DRY=1; YES=1 ;;  # auto-approve prompts, just show what would happen
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

run() {
    # run "label" "command" [args...]
    local label="$1"; shift
    if [ "$DRY" = "1" ]; then
        echo "[dry-run] would $label: $*"
    else
        "$@" 2>/dev/null || true
        echo "  $label"
    fi
}

ask() {
    if [ "$YES" = "1" ] || [ "$FORCE" = "1" ]; then
        echo "[auto] $1"
        return 0
    fi
    read -rp "$1 [Y/n] " a
    [ -z "$a" ] && return 0
    [[ "$a" =~ ^[Yy] ]] && return 0
    return 1
}

echo "============================================"
echo "  Luna Reset Tool"
echo "============================================"
echo ""

# 1. Stop and remove containers
echo "--- 1. Containers ---"
for name in luna-tts luna-llama; do
    if podman ps -a --filter "name=$name" --format "{{.Names}}" 2>/dev/null | grep -q "$name"; then
        ask "Stop and remove container '$name'?"
        run "stop container $name" podman stop --time 10 "$name"
        run "remove container $name" podman rm -f "$name"
        echo "  removed container: $name"
    else
        echo "  skipped (not found): $name"
    fi
done
echo ""

# 2. Remove Luna-owned images
echo "--- 2. Podman Images ---"
luna_images=""
luna_images=$(podman images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | grep "^localhost/luna-" || true) || true
if [ -n "$luna_images" ]; then
    ask "Remove Luna Podman images?"
    while IFS= read -r img; do
        run "remove image $img" podman rmi -f "$img"
        echo "  removed image: $img"
    done <<< "$luna_images"
else
    echo "  skipped (no Luna images found)"
fi

# Prune dangling images (unlabeled build layers)
ask "Prune dangling (unlabeled) Podman images?"
run "prune dangling Podman images" podman image prune -f
echo "  pruned dangling images"
echo ""

# 3. Remove virtual environment
echo "--- 3. Virtual Environment ---"
venv_dir="$ROOT/.venv"
if [ -d "$venv_dir" ]; then
    ask "Remove virtual environment ($venv_dir)?"
    if [ "$DRY" = "1" ]; then
        echo "[dry-run] would remove venv: $venv_dir"
    else
        # Use find to delete contents first (handles NFS-stale locks), then the dir
        find "$venv_dir" -mindepth 1 -type f -delete 2>/dev/null || true
        find "$venv_dir" -mindepth 1 -type d -empty -delete 2>/dev/null || true
        rm -rf "$venv_dir" 2>/dev/null || true
        # Last resort: force-unmount if it's a mount point
        if mountpoint -q "$venv_dir" 2>/dev/null; then
            umount -l "$venv_dir" 2>/dev/null || true
            rm -rf "$venv_dir" 2>/dev/null || true
        fi
    fi
    echo "  removed venv"
else
    echo "  skipped (not found): $venv_dir"
fi
echo ""

# 4. Remove Python cache files
echo "--- 4. Python Cache ---"
ask "Remove __pycache__ directories?"
if [ "$DRY" = "1" ]; then
    count=$(find "$ROOT" -type d -name "__pycache__" 2>/dev/null | wc -l)
    echo "[dry-run] would remove $count __pycache__ directories"
else
    find "$ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
fi
echo "  removed __pycache__"
echo ""

# 5. Remove LLM models (if --all or explicit)
echo "--- 5. LLM Models ---"
model_dir="$ROOT/llama/models"
if [ -d "$model_dir" ] && [ "$(ls -A "$model_dir" 2>/dev/null)" ]; then
    if [ "$ALL" = "1" ]; then
        ask "Remove LLM models ($model_dir, ~2-6 GB)?"
        if [ "$DRY" = "1" ]; then
            echo "[dry-run] would remove models: $model_dir"
        else
            rm -rf "$model_dir"
        fi
        echo "  removed models"
    else
        echo "  skipped (use --all to remove models)"
    fi
else
    echo "  skipped (not found)"
fi
echo ""

# 6. Remove llama container config files
echo "--- 6. Container Config ---"
ask "Remove container data directory (llama/data/)?"
if [ -d "$ROOT/llama/data" ]; then
    if [ "$DRY" = "1" ]; then
        echo "[dry-run] would remove directory: $ROOT/llama/data"
    else
        rm -rf "$ROOT/llama/data"
    fi
    echo "  removed llama/data/"
fi
echo ""

# 7. Remove install metadata
echo "--- 7. Install Metadata ---"
if [ -f "$ROOT/install.json" ]; then
    ask "Remove install.json (installation record)?"
    if [ "$DRY" = "1" ]; then
        echo "[dry-run] would remove file: $ROOT/install.json"
    else
        rm -f "$ROOT/install.json"
    fi
    echo "  removed install.json"
fi
echo ""

# 8. Remove launcher files
echo "--- 8. Launchers ---"
ask "Remove launcher files (luna command, .desktop, .cmd, .command)?"
# Remove luna command if it points into this project
luna_cmd="$HOME/.local/bin/luna"
if [ -L "$luna_cmd" ] || [ -f "$luna_cmd" ]; then
    content=$(cat "$luna_cmd" 2>/dev/null || true)
    if echo "$content" | grep -q "ai-voice\|$ROOT"; then
        if [ "$DRY" = "1" ]; then
            echo "[dry-run] would remove launcher: $luna_cmd"
        else
            rm -f "$luna_cmd"
        fi
        echo "  removed launcher: $luna_cmd"
    else
        echo "  skipped (not Luna-owned): $luna_cmd"
    fi
fi
if [ "$DRY" = "1" ]; then
    [ -f "$ROOT/luna.cmd" ] && echo "[dry-run] would remove file: $ROOT/luna.cmd"
    [ -f "$ROOT/Luna.command" ] && echo "[dry-run] would remove file: $ROOT/Luna.command"
    [ -f "$HOME/.local/share/applications/luna.desktop" ] && echo "[dry-run] would remove file: $HOME/.local/share/applications/luna.desktop"
else
    rm -f "$ROOT/luna.cmd" "$ROOT/Luna.command"
    if [ -f "$HOME/.local/share/applications/luna.desktop" ]; then
        rm -f "$HOME/.local/share/applications/luna.desktop"
    fi
fi
echo "  removed launchers"
echo ""

# 9. Remove config.json (optional, use --all)
echo "--- 9. Config ---"
if [ "$ALL" = "1" ] && [ -f "$ROOT/config.json" ]; then
    ask "Remove config.json (voice, STT, TTS settings)?"
    if [ "$DRY" = "1" ]; then
        echo "[dry-run] would remove file: $ROOT/config.json"
    else
        rm -f "$ROOT/config.json"
    fi
    echo "  removed config.json"
else
    echo "  skipped (config.json preserved; use --all to remove)"
fi
echo ""

# 10. System packages (very optional, use --all and explicit confirm)
echo "--- 10. System Packages ---"
if [ "$ALL" = "1" ]; then
    ask "Remove system packages installed by install.sh (vulkan-headers, glslc, cmake, etc.)? WARNING: may break other software."
    if [ -f "$ROOT/install.sh" ]; then
        # The install.sh may have logged what it installed; re-install script
        # doesn't track this, so we can't auto-rollback. Skip.
        echo "  skipped (system package removal is destructive and not reversible)"
    fi
fi
echo ""

echo "============================================"
echo "  Reset complete."
echo "============================================"
echo ""
echo "Your data (config.json, memory/, history/, agent/) is preserved."
echo "To start fresh: run ./installer/install.sh or ./installer/install.py"
