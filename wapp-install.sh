#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: wapp-install <pkg-name> \"<Display Name>\" <app-dir-or-exe> [exe-name-if-dir]"
    echo ""
    echo "Examples:"
    echo "  Portable Folder: wapp-install flstudio \"FL Studio\" ./FLStudio_Dir FL64.exe"
    echo "  Single EXE:      wapp-install Rufus \"Rufus\" ./rufus.exe"
    exit 1
}

if [[ $# -lt 3 ]]; then
    usage
fi

# Sanitize package name for pacman (lowercase, alphanumeric + hyphens)
RAW_PKG="$1"
PKG_NAME="wapp-$(echo "$RAW_PKG" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g')"
APP_ID="$(echo "$RAW_PKG" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g')"
DISPLAY_NAME="$2"
SOURCE_PATH="$(realpath "$3")"
EXE_REL="${4:-}"

if [[ ! -e "$SOURCE_PATH" ]]; then
    echo "Error: Source path '$SOURCE_PATH' does not exist."
    exit 1
fi

if [[ -d "$SOURCE_PATH" && -z "$EXE_REL" ]]; then
    echo "Error: When passing a directory, you must specify the relative executable path as the 4th argument."
    usage
fi

if [[ -f "$SOURCE_PATH" ]]; then
    EXE_REL="$(basename "$SOURCE_PATH")"
fi

# Create temporary build workspace
BUILD_DIR="$(mktemp -d -t wapp-build-XXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT

mkdir -p "$BUILD_DIR/app_payload"

echo "==> Copying application files to build workspace..."
if [[ -d "$SOURCE_PATH" ]]; then
    cp -a "$SOURCE_PATH/." "$BUILD_DIR/app_payload/"
else
    cp -a "$SOURCE_PATH" "$BUILD_DIR/app_payload/"
fi

TARGET_EXE="$BUILD_DIR/app_payload/$EXE_REL"
if [[ ! -f "$TARGET_EXE" ]]; then
    echo "Error: Could not find executable '$EXE_REL' inside the application payload."
    exit 1
fi

# Attempt to extract icon from the Windows executable
HAS_ICON=0
if command -v wrestool >/dev/null 2>&1 && command -v magick >/dev/null 2>&1; then
    echo "==> Extracting icon from $EXE_REL..."
    if wrestool -x -t 14 "$TARGET_EXE" > "$BUILD_DIR/raw.ico" 2>/dev/null && [[ -s "$BUILD_DIR/raw.ico" ]]; then
        # Convert the highest resolution layer to PNG
        magick "$BUILD_DIR/raw.ico[-1]" "$BUILD_DIR/$APP_ID.png" 2>/dev/null || true
        if [[ -f "$BUILD_DIR/$APP_ID.png" ]]; then
            HAS_ICON=1
        fi
    fi
fi

# 1. Generate the Wrapper Script
cat <<EOF > "$BUILD_DIR/launcher.sh"
#!/usr/bin/env bash
set -e

APP_ID="$APP_ID"
SYS_APP_DIR="/opt/wapps/\$APP_ID"
export WINEPREFIX="\$HOME/.wine-prefixes/\$APP_ID"
DEST_DIR="\$WINEPREFIX/drive_c/\$APP_ID"

# Allow user to nuke/reset the prefix if needed
if [[ "\${1:-}" == "--reset-prefix" ]]; then
    echo "Removing prefix \$WINEPREFIX..."
    rm -rf "\$WINEPREFIX"
    shift
fi

# Bootstrap prefix and copy app contents on first run
if [[ ! -d "\$DEST_DIR" ]]; then
    echo "Initializing isolated Wine prefix at \$WINEPREFIX..."
    mkdir -p "\$WINEPREFIX"
    wineboot -u
    mkdir -p "\$DEST_DIR"
    echo "Copying application contents into prefix..."
    cp -a "\$SYS_APP_DIR/." "\$DEST_DIR/"
fi

cd "\$DEST_DIR"
exec wine "\$DEST_DIR/$EXE_REL" "\$@"
EOF

# 2. Generate the Desktop Entry
ICON_VAL="wine"
if [[ $HAS_ICON -eq 1 ]]; then
    ICON_VAL="$APP_ID"
fi

cat <<EOF > "$BUILD_DIR/$APP_ID.desktop"
[Desktop Entry]
Type=Application
Name=$DISPLAY_NAME
Comment=$DISPLAY_NAME (Isolated Wine Application)
Exec=/usr/bin/$PKG_NAME %F
Icon=$ICON_VAL
Terminal=false
Categories=Wine;Application;
StartupNotify=true
EOF

# 3. Generate the PKGBUILD
cat <<EOF > "$BUILD_DIR/PKGBUILD"
# Maintainer: Local User (generated via wapp-install)
pkgname=$PKG_NAME
pkgver=1.0.0
pkgrel=1
pkgdesc="$DISPLAY_NAME (Wine App Wrapper)"
arch=('any')
depends=('wine')
options=('!strip')

package() {
    # Install raw app files to /opt/wapps/<id>
    install -d "\$pkgdir/opt/wapps/$APP_ID"
    cp -a "\$srcdir/../app_payload/." "\$pkgdir/opt/wapps/$APP_ID/"

    # Install wrapper launcher to /usr/bin
    install -Dm755 "\$srcdir/../launcher.sh" "\$pkgdir/usr/bin/$PKG_NAME"

    # Install desktop entry
    install -Dm644 "\$srcdir/../$APP_ID.desktop" "\$pkgdir/usr/share/applications/$APP_ID.desktop"

    # Install extracted icon if available
    if [[ -f "\$srcdir/../$APP_ID.png" ]]; then
        install -Dm644 "\$srcdir/../$APP_ID.png" "\$pkgdir/usr/share/pixmaps/$APP_ID.png"
    fi
}
EOF

echo "==> Building and installing pacman package '$PKG_NAME'..."
cd "$BUILD_DIR"
makepkg -si --needed
echo "==> Done! '$DISPLAY_NAME' is now installed as package '$PKG_NAME' and available in your application menu."
