#!/usr/bin/env bash
#
# Vercel build: install the Rust toolchain + wasm-pack (neither ships in Vercel's
# build image), then run the web build. `npm run build` fires the `prebuild`
# hook, which runs `wasm-pack build` against crates/app_wasm into web/pkg before
# `tsc && vite build`.
set -euo pipefail

WASM_PACK_VERSION="0.13.1"

echo "==> Installing Rust toolchain"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
fi
export PATH="$HOME/.cargo/bin:$PATH"
# shellcheck disable=SC1091
source "$HOME/.cargo/env" 2>/dev/null || true

rustup target add wasm32-unknown-unknown

echo "==> Installing wasm-pack ${WASM_PACK_VERSION}"
if ! command -v wasm-pack >/dev/null 2>&1; then
  WP="wasm-pack-v${WASM_PACK_VERSION}-x86_64-unknown-linux-musl"
  curl -sSL "https://github.com/rustwasm/wasm-pack/releases/download/v${WASM_PACK_VERSION}/${WP}.tar.gz" \
    | tar -xz -C /tmp
  export PATH="/tmp/${WP}:$PATH"
fi

echo "==> wasm-pack $(wasm-pack --version) / $(rustc --version)"
echo "==> Building web (wasm-pack + vite)"
npm --prefix web run build
