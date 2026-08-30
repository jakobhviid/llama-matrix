#!/bin/sh
# llama-matrix installer - no Homebrew, no compiler, no root. Downloads the
# prebuilt binary for your OS/arch into a bin dir on your PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/jakobhviid/llama-matrix/main/install.sh | sh
#
# Override the install dir with LLAMA_MATRIX_BIN_DIR.
set -eu

REPO="jakobhviid/llama-matrix"
NAME="llama-matrix"
BIN_DIR="${LLAMA_MATRIX_BIN_DIR:-$HOME/.local/bin}"

os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
    Linux)  target_os="unknown-linux-musl" ;;
    Darwin) target_os="apple-darwin" ;;
    *) echo "unsupported OS: $os" >&2; exit 1 ;;
esac

case "$arch" in
    x86_64 | amd64)   target_arch="x86_64" ;;
    aarch64 | arm64)  target_arch="aarch64" ;;
    *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
esac

asset="${NAME}-${target_arch}-${target_os}.tar.gz"
url="https://github.com/${REPO}/releases/latest/download/${asset}"

echo "Installing ${NAME} (${target_arch}-${target_os}) → ${BIN_DIR}"
mkdir -p "$BIN_DIR"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if ! curl -fsSL "$url" | tar xz -C "$tmp"; then
    echo "download/extract failed: $url" >&2
    exit 1
fi

installed=""
for f in "$tmp"/*; do
    [ -f "$f" ] || continue
    name="$(basename "$f")"
    install -m 0755 "$f" "$BIN_DIR/$name"
    installed="${installed} ${name}"
done

echo "Installed:${installed}"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "note: ${BIN_DIR} is not on your PATH - add it, e.g.:"
       echo "      export PATH=\"${BIN_DIR}:\$PATH\"" ;;
esac

echo "Done. Run \`llama-matrix --help\` (or \`llama-matrix setup\` to get started)."
