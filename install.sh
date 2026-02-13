#!/bin/sh
set -eu

# harkd installer
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/harkhq/harkd/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/harkhq/harkd/main/install.sh | sh -s -- --uninstall

REPO="git+https://github.com/harkhq/harkd"
SERVICE_NAME="harkd"
LAUNCHD_LABEL="com.harkhq.harkd"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { printf '  \033[1;34m→\033[0m %s\n' "$*"; }
ok()    { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[1;33m!\033[0m %s\n' "$*" >&2; }
err()   { printf '  \033[1;31m✗\033[0m %s\n' "$*" >&2; }
die()   { err "$@"; exit 1; }

has() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# Detect OS / distro / package manager
# ---------------------------------------------------------------------------

detect_platform() {
    OS="$(uname -s)"
    case "$OS" in
        Linux)
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                DISTRO="${ID:-linux}"
            else
                DISTRO="linux"
            fi

            if has apt-get; then
                PKG_MGR="apt"
            elif has dnf; then
                PKG_MGR="dnf"
            elif has pacman; then
                PKG_MGR="pacman"
            elif has zypper; then
                PKG_MGR="zypper"
            else
                die "Unsupported Linux distribution — no recognised package manager found"
            fi
            ;;
        Darwin)
            DISTRO="macos"
            has brew || die "Homebrew is required on macOS. Install it from https://brew.sh"
            PKG_MGR="brew"
            ;;
        *)
            die "Unsupported operating system: $OS"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------

install_sys_deps() {
    info "Checking system dependencies (portaudio, ffmpeg)..."

    need_portaudio=false
    need_ffmpeg=false

    # portaudio: check for shared lib or dev headers
    if ! has pkg-config || ! pkg-config --exists portaudio-2.0 2>/dev/null; then
        case "$PKG_MGR" in
            brew)
                if ! brew list portaudio >/dev/null 2>&1; then
                    need_portaudio=true
                fi
                ;;
            *)  need_portaudio=true ;;
        esac
    fi

    has ffmpeg || need_ffmpeg=true

    if [ "$need_portaudio" = false ] && [ "$need_ffmpeg" = false ]; then
        ok "System dependencies already installed"
        return
    fi

    pkgs=""
    case "$PKG_MGR" in
        apt)
            $need_portaudio && pkgs="$pkgs portaudio19-dev"
            $need_ffmpeg    && pkgs="$pkgs ffmpeg"
            ;;
        dnf)
            $need_portaudio && pkgs="$pkgs portaudio-devel"
            $need_ffmpeg    && pkgs="$pkgs ffmpeg-free"
            ;;
        pacman)
            $need_portaudio && pkgs="$pkgs portaudio"
            $need_ffmpeg    && pkgs="$pkgs ffmpeg"
            ;;
        zypper)
            $need_portaudio && pkgs="$pkgs portaudio-devel"
            $need_ffmpeg    && pkgs="$pkgs ffmpeg"
            ;;
        brew)
            $need_portaudio && pkgs="$pkgs portaudio"
            $need_ffmpeg    && pkgs="$pkgs ffmpeg"
            ;;
    esac

    # shellcheck disable=SC2086
    case "$PKG_MGR" in
        apt)    info "Installing: $pkgs"; sudo apt-get update -qq && sudo apt-get install -y -qq $pkgs ;;
        dnf)    info "Installing: $pkgs"; sudo dnf install -y -q $pkgs ;;
        pacman) info "Installing: $pkgs"; sudo pacman -S --noconfirm --needed $pkgs ;;
        zypper) info "Installing: $pkgs"; sudo zypper install -y $pkgs ;;
        brew)   info "Installing: $pkgs"; brew install $pkgs ;;
    esac

    ok "System dependencies installed"
}

# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------

install_uv() {
    if has uv; then
        ok "uv already installed"
        return
    fi

    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source uv's env so it's available in this session
    UV_ENV="$HOME/.local/bin/env"
    if [ -f "$UV_ENV" ]; then
        # shellcheck disable=SC1090
        . "$UV_ENV"
    fi

    # Also try cargo env (alternative install location)
    CARGO_ENV="$HOME/.cargo/env"
    if [ -f "$CARGO_ENV" ]; then
        # shellcheck disable=SC1090
        . "$CARGO_ENV"
    fi

    has uv || die "uv installed but not found on PATH. Restart your shell and re-run this script."
    ok "uv installed"
}

# ---------------------------------------------------------------------------
# harkd
# ---------------------------------------------------------------------------

install_harkd() {
    info "Installing harkd..."

    uv tool install --force --python '>=3.11' "harkd @ $REPO"

    UV_BIN_DIR="$(uv tool dir --bin)"
    HARKD_BIN="$UV_BIN_DIR/harkd"

    if [ ! -f "$HARKD_BIN" ]; then
        die "Installation succeeded but harkd binary not found at $HARKD_BIN"
    fi

    ok "harkd installed at $HARKD_BIN"
}

# ---------------------------------------------------------------------------
# Service registration
# ---------------------------------------------------------------------------

register_service_linux() {
    UV_BIN_DIR="$(uv tool dir --bin)"
    HARKD_BIN="$UV_BIN_DIR/harkd"

    SERVICE_DIR="$HOME/.config/systemd/user"
    SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME.service"

    mkdir -p "$SERVICE_DIR"

    cat > "$SERVICE_FILE" <<UNIT
[Unit]
Description=Hark voice recording and transcription daemon
After=network.target pipewire.service pulseaudio.service

[Service]
Type=simple
ExecStart=$HARKD_BIN
Restart=on-failure
RestartSec=5
Environment=PATH=$UV_BIN_DIR:/usr/local/bin:/usr/bin:/bin
StandardOutput=journal
StandardError=journal
SyslogIdentifier=harkd
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=%h/.local/share/hark %h/.config/hark %h/.cache
PrivateTmp=true

[Install]
WantedBy=default.target
UNIT

    systemctl --user daemon-reload
    systemctl --user enable --now "$SERVICE_NAME"

    ok "systemd user service enabled and started"

    # Enable lingering so the service starts at boot (non-fatal)
    if has loginctl; then
        if loginctl enable-linger 2>/dev/null; then
            ok "Login lingering enabled (service will start at boot)"
        else
            warn "Could not enable login lingering — run: sudo loginctl enable-linger $(whoami)"
        fi
    fi
}

register_service_macos() {
    UV_BIN_DIR="$(uv tool dir --bin)"
    HARKD_BIN="$UV_BIN_DIR/harkd"

    AGENT_DIR="$HOME/Library/LaunchAgents"
    PLIST_FILE="$AGENT_DIR/$LAUNCHD_LABEL.plist"
    LOG_DIR="$HOME/Library/Logs/harkd"

    mkdir -p "$AGENT_DIR" "$LOG_DIR"

    cat > "$PLIST_FILE" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LAUNCHD_LABEL</string>

    <key>ProgramArguments</key>
    <array>
        <string>$HARKD_BIN</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$UV_BIN_DIR:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>5</integer>

    <key>StandardOutPath</key>
    <string>$LOG_DIR/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>$LOG_DIR/stderr.log</string>
</dict>
</plist>
PLIST

    # Idempotent: unload first if already loaded
    launchctl bootout "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_FILE"

    ok "launchd agent loaded and started"
}

register_service() {
    info "Registering background service..."

    case "$OS" in
        Linux)  register_service_linux ;;
        Darwin) register_service_macos ;;
    esac
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

uninstall() {
    info "Uninstalling harkd..."

    # Stop and remove service
    case "$(uname -s)" in
        Linux)
            SERVICE_FILE="$HOME/.config/systemd/user/$SERVICE_NAME.service"
            if [ -f "$SERVICE_FILE" ]; then
                systemctl --user disable --now "$SERVICE_NAME" 2>/dev/null || true
                rm -f "$SERVICE_FILE"
                systemctl --user daemon-reload
                ok "systemd service removed"
            else
                warn "systemd service file not found — skipping"
            fi
            ;;
        Darwin)
            PLIST_FILE="$HOME/Library/LaunchAgents/$LAUNCHD_LABEL.plist"
            if [ -f "$PLIST_FILE" ]; then
                launchctl bootout "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || true
                rm -f "$PLIST_FILE"
                ok "launchd agent removed"
            else
                warn "launchd plist not found — skipping"
            fi
            ;;
    esac

    # Remove harkd via uv
    if has uv; then
        uv tool uninstall harkd 2>/dev/null && ok "harkd uninstalled" || warn "harkd was not installed via uv"
    else
        warn "uv not found — skipping tool uninstall"
    fi

    echo ""
    info "Data directories preserved:"
    info "  ~/.local/share/hark/  (recordings & transcriptions)"
    info "  ~/.config/hark/       (configuration)"
    info "Remove them manually if you no longer need them."

    ok "Uninstall complete"
    exit 0
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
    UV_BIN_DIR="$(uv tool dir --bin)"

    echo ""
    echo "  ╔══════════════════════════════════════════════╗"
    echo "  ║           harkd installed successfully       ║"
    echo "  ╚══════════════════════════════════════════════╝"
    echo ""
    info "API:          http://localhost:8765"
    info "Health check: curl -s http://localhost:8765/health"
    echo ""
    case "$OS" in
        Linux)
            info "Service management:"
            info "  systemctl --user status harkd    # check status"
            info "  journalctl --user -u harkd -f    # view logs"
            info "  systemctl --user restart harkd   # restart"
            info "  systemctl --user stop harkd      # stop"
            ;;
        Darwin)
            info "Service management:"
            info "  launchctl print gui/$(id -u)/$LAUNCHD_LABEL   # check status"
            info "  tail -f ~/Library/Logs/harkd/stderr.log       # view logs"
            info "  launchctl kickstart -k gui/$(id -u)/$LAUNCHD_LABEL  # restart"
            info "  launchctl kill SIGTERM gui/$(id -u)/$LAUNCHD_LABEL  # stop"
            ;;
    esac
    echo ""
    info "Uninstall:"
    info "  curl -fsSL https://raw.githubusercontent.com/harkhq/harkd/main/install.sh | sh -s -- --uninstall"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo ""
    echo "  harkd installer"
    echo "  ───────────────"
    echo ""

    # Handle --uninstall flag
    for arg in "$@"; do
        case "$arg" in
            --uninstall) uninstall ;;
        esac
    done

    detect_platform
    install_sys_deps
    install_uv
    install_harkd
    register_service
    print_summary
}

main "$@"
