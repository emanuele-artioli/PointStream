#!/usr/bin/env bash
# Installed as ~/.pointstream-runtime/bootstrap-hostlocal.sh; sourced by .bashrc/.profile.
# Provision runtime aliases on every host before installation (docs/fleet.md).
PS_RUNTIME_ROOT="/local/users/$(id -un)/runtime"
if [ ! -d "${PS_RUNTIME_ROOT%/runtime}" ] || \
   [ "$(stat -c %d "${PS_RUNTIME_ROOT%/runtime}" 2>/dev/null)" = "$(stat -c %d / 2>/dev/null)" ]; then
    printf '%s\n' 'PointStream: separate /local user storage is unavailable; refusing root-disk runtime setup.' >&2
    return 1
fi
HOSTLOCAL_SERVERS="$PS_RUNTIME_ROOT/editor-servers"
HOSTLOCAL_PYCACHE="$PS_RUNTIME_ROOT/pycache"
HOSTLOCAL_CODEX="$PS_RUNTIME_ROOT/codex"
HOSTLOCAL_CODEX_BIN="$PS_RUNTIME_ROOT/codex-bin"
mkdir -p "$HOSTLOCAL_CODEX" "$HOSTLOCAL_CODEX_BIN" "$HOSTLOCAL_SERVERS/cursor-server" \
    "$HOSTLOCAL_SERVERS/vscode-server" "$HOSTLOCAL_PYCACHE" "$PS_RUNTIME_ROOT/tmp" "$PS_RUNTIME_ROOT/agentctl" || return 1
chmod 700 "$PS_RUNTIME_ROOT" "$HOSTLOCAL_CODEX" "$HOSTLOCAL_CODEX_BIN" \
    "$HOSTLOCAL_SERVERS" "$HOSTLOCAL_PYCACHE" "$PS_RUNTIME_ROOT/tmp" "$PS_RUNTIME_ROOT/agentctl"
# Keep durable preferences shared and runtime databases host-local.
if [ -f "$HOME/.codex/config.toml" ] && { [ ! -e "$HOSTLOCAL_CODEX/config.toml" ] || [ "$HOME/.codex/config.toml" -nt "$HOSTLOCAL_CODEX/config.toml" ]; }; then
    install -m 600 "$HOME/.codex/config.toml" "$HOSTLOCAL_CODEX/config.toml"
fi
if [ ! -e "$HOSTLOCAL_CODEX/auth.json" ] && [ -f "$HOME/.codex/auth.json" ]; then
    install -m 600 "$HOME/.codex/auth.json" "$HOSTLOCAL_CODEX/auth.json"
fi
if [ ! -e "$HOSTLOCAL_CODEX_BIN/codex" ] && [ -x "$HOME/.local/bin/codex" ]; then
    ln -s "$(readlink -f "$HOME/.local/bin/codex")" "$HOSTLOCAL_CODEX_BIN/codex"
fi
export CODEX_HOME="$HOSTLOCAL_CODEX" CODEX_SQLITE_HOME="$HOSTLOCAL_CODEX" CODEX_INSTALL_DIR="$HOSTLOCAL_CODEX_BIN"
case ":$PATH:" in *":$HOSTLOCAL_CODEX_BIN:"*) ;; *) export PATH="$HOSTLOCAL_CODEX_BIN:$PATH" ;; esac
export PYTHONPYCACHEPREFIX="$HOSTLOCAL_PYCACHE" PYTHONNOUSERSITE=1
export TMPDIR="$PS_RUNTIME_ROOT/tmp" TMP="$PS_RUNTIME_ROOT/tmp" TEMP="$PS_RUNTIME_ROOT/tmp"
export AGENTCTL_RUNTIME="$PS_RUNTIME_ROOT/agentctl"
export PS_LOCAL_ROOT="/local/users/$(id -un)/pointstream"
