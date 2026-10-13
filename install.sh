#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VERSION="0.2.0"
OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH_NAME="$(uname -m)"

MODE="install"
SKIP_MODELS=0

for arg in "$@"; do
  case "$arg" in
    --bundle)
      MODE="bundle"
      ;;
    --all)
      MODE="all"
      ;;
    --skip-models)
      SKIP_MODELS=1
      ;;
    -h|--help)
      cat <<EOF
Prism v${VERSION} — Single-Script Installer & Bundler (Linux & macOS)

Usage:
  ./install.sh                 Build and install Prism CLI + Desktop GUI for the current user
  ./install.sh --bundle        Build and package a self-contained distributable bundle in dist/
  ./install.sh --all           Both install locally and generate the distributable bundle in dist/
  ./install.sh --skip-models   Skip pre-downloading EmbeddingGemma 2 GGUF models during install
EOF
      exit 0
      ;;
  esac
done

echo "================================================================"
echo "  Prism v${VERSION} — Multimodal Search, Semantic ASR & Verifier"
echo "  Platform : ${OS_NAME} (${ARCH_NAME})"
echo "  Mode     : ${MODE}"
echo "================================================================"

# 1. Ensure Vendor/pdfium is present
if [[ ! -d "$ROOT_DIR/Vendor/pdfium/lib" ]]; then
  echo "[1/5] Fetching prebuilt PDFium library..."
  bash "$ROOT_DIR/scripts/fetch-vendor.sh"
else
  echo "[1/5] PDFium library ready in Vendor/pdfium/lib"
fi

# 2. Ensure Vendor/llama (patched llama.cpp b11461 + mtmd) is built
if [[ ! -d "$ROOT_DIR/Vendor/llama/lib" ]]; then
  echo "[2/5] Building patched llama.cpp + mtmd for EmbeddingGemma 2..."
  bash "$ROOT_DIR/scripts/build-llama.sh"
else
  echo "[2/5] Patched llama.cpp + mtmd ready in Vendor/llama/lib"
fi

# 3. Build Release Binaries (prism CLI + prism-app GUI)
echo "[3/5] Compiling prism (CLI) and prism-app (Native Desktop GUI) in release mode..."
cargo build --release

PRISM_BIN="$ROOT_DIR/target/release/prism"
PRISM_APP_BIN="$ROOT_DIR/target/release/prism-app"

if [[ ! -f "$PRISM_BIN" || ! -f "$PRISM_APP_BIN" ]]; then
  echo "Error: Release binaries were not produced in target/release/" >&2
  exit 1
fi

install_local() {
  local bin_dir="$HOME/.local/bin"
  local lib_dir="$HOME/.local/lib/prism"
  mkdir -p "$bin_dir" "$lib_dir"

  echo "[4/5] Installing binaries to $bin_dir and shared libraries to $lib_dir ..."
  cp -f "$PRISM_BIN" "$bin_dir/prism"
  cp -f "$PRISM_APP_BIN" "$bin_dir/prism-app"
  chmod +x "$bin_dir/prism" "$bin_dir/prism-app"

  cp -a "$ROOT_DIR/Vendor/llama/lib/"* "$lib_dir/" 2>/dev/null || true
  cp -a "$ROOT_DIR/Vendor/pdfium/lib/"* "$lib_dir/" 2>/dev/null || true

  if [[ "$OS_NAME" == "linux" ]]; then
    local app_dir="$HOME/.local/share/applications"
    local icon_dir="$HOME/.local/share/icons/hicolor/256x256/apps"
    mkdir -p "$app_dir" "$icon_dir"
    cp -f "$ROOT_DIR/crates/prism-app/src-tauri/icons/icon.png" "$icon_dir/prism.png"

    cat > "$app_dir/prism.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Prism
GenericName=Multimodal Search & Semantic ASR
Comment=Search files by text, spoken audio, and verify part-of-file containment with EmbeddingGemma 2
Exec=$bin_dir/prism-app %F
Icon=$icon_dir/prism.png
Terminal=false
Categories=Utility;Search;AudioVideo;Development;
Keywords=search;multimodal;asr;audio;video;pdf;embeddinggemma;
StartupWMClass=prism-app
EOF
    chmod +x "$app_dir/prism.desktop"
    echo "      Installed Linux desktop launcher: $app_dir/prism.desktop"
  elif [[ "$OS_NAME" == "darwin" ]]; then
    local mac_app="$HOME/Applications/Prism.app"
    mkdir -p "$mac_app/Contents/MacOS" "$mac_app/Contents/Frameworks" "$mac_app/Contents/Resources"
    cp -f "$PRISM_APP_BIN" "$mac_app/Contents/MacOS/prism-app"
    cp -f "$PRISM_BIN" "$mac_app/Contents/MacOS/prism"
    cp -a "$lib_dir/"* "$mac_app/Contents/Frameworks/" 2>/dev/null || true
    cp -f "$ROOT_DIR/crates/prism-app/src-tauri/icons/icon.png" "$mac_app/Contents/Resources/icon.png"
    cat > "$mac_app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>Prism</string>
  <key>CFBundleDisplayName</key>
  <string>Prism</string>
  <key>CFBundleIdentifier</key>
  <string>dev.rabimba.prism</string>
  <key>CFBundleVersion</key>
  <string>${VERSION}</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION}</string>
  <key>CFBundleExecutable</key>
  <string>prism-app</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF
    echo "      Installed macOS application bundle: $mac_app"
  fi

  if [[ "$SKIP_MODELS" -eq 0 ]]; then
    echo "[5/5] Ensuring EmbeddingGemma 2 GGUF models are downloaded and verified..."
    "$bin_dir/prism" download-model
  else
    echo "[5/5] Skipped model download (--skip-models)."
  fi
}

build_bundle() {
  local bundle_name="prism-v${VERSION}-${OS_NAME}-${ARCH_NAME}"
  local dist_root="$ROOT_DIR/dist"
  local stage_dir="$dist_root/$bundle_name"

  rm -rf "$stage_dir"
  mkdir -p "$stage_dir/bin" "$stage_dir/lib/prism" "$stage_dir/share/icons"

  echo "[Bundle] Packaging portable distribution into $stage_dir ..."
  cp -f "$PRISM_BIN" "$stage_dir/bin/prism"
  cp -f "$PRISM_APP_BIN" "$stage_dir/bin/prism-app"
  chmod +x "$stage_dir/bin/prism" "$stage_dir/bin/prism-app"

  cp -a "$ROOT_DIR/Vendor/llama/lib/"* "$stage_dir/lib/prism/" 2>/dev/null || true
  cp -a "$ROOT_DIR/Vendor/pdfium/lib/"* "$stage_dir/lib/prism/" 2>/dev/null || true
  cp -f "$ROOT_DIR/crates/prism-app/src-tauri/icons/icon.png" "$stage_dir/share/icons/prism.png"

  cat > "$stage_dir/install-bundle.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HOME/.local/bin" "$HOME/.local/lib/prism" "$HOME/.local/share/applications" "$HOME/.local/share/icons/hicolor/256x256/apps"
cp -f "$BUNDLE_DIR/bin/prism" "$HOME/.local/bin/prism"
cp -f "$BUNDLE_DIR/bin/prism-app" "$HOME/.local/bin/prism-app"
chmod +x "$HOME/.local/bin/prism" "$HOME/.local/bin/prism-app"
cp -a "$BUNDLE_DIR/lib/prism/"* "$HOME/.local/lib/prism/"
cp -f "$BUNDLE_DIR/share/icons/prism.png" "$HOME/.local/share/icons/hicolor/256x256/apps/prism.png"

if [[ "$(uname -s)" == "Linux" ]]; then
  cat > "$HOME/.local/share/applications/prism.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Prism
GenericName=Multimodal Search & Semantic ASR
Comment=Search files by text, spoken audio, and verify part-of-file containment with EmbeddingGemma 2
Exec=$HOME/.local/bin/prism-app %F
Icon=$HOME/.local/share/icons/hicolor/256x256/apps/prism.png
Terminal=false
Categories=Utility;Search;AudioVideo;Development;
StartupWMClass=prism-app
DESKTOP
  chmod +x "$HOME/.local/share/applications/prism.desktop"
fi

echo "Prism installed to ~/.local/bin/prism and ~/.local/bin/prism-app!"
EOF
  chmod +x "$stage_dir/install-bundle.sh"

  tar -C "$dist_root" -czf "$dist_root/${bundle_name}.tar.gz" "$bundle_name"
  echo "[Bundle] Created distributable archive: $dist_root/${bundle_name}.tar.gz"
}

if [[ "$MODE" == "install" ]]; then
  install_local
elif [[ "$MODE" == "bundle" ]]; then
  build_bundle
elif [[ "$MODE" == "all" ]]; then
  install_local
  build_bundle
fi

echo ""
echo "✅ Prism v${VERSION} setup complete!"
echo "   - Launch Desktop GUI : prism gui   (or: prism-app)"
echo "   - CLI Search         : prism search \"your query\" -d /path/to/folder"
echo "   - Spoken Audio Search: prism search --audio query.wav -d /path/to/folder"
echo "   - Part-of-File Check : prism contain candidate_clip.mp4 -d /path/to/folder"
