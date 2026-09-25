#!/usr/bin/env sh
# Assemble the static web app in site/ (run by Vercel, see vercel.json; also works locally).
#   - copies the figures from docs/images
#   - fetches the pinned onnxruntime-web runtime and verifies its SHA-256 checksums
#     (served same-origin so multi-threaded WebAssembly works under COOP/COEP)
# Local preview:  sh scripts/build_site.sh && python -m http.server -d site 8000
set -eu

ORT_VERSION=1.30.0
ORT_URL="https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$ROOT/site"

mkdir -p "$SITE/images" "$SITE/vendor/ort-${ORT_VERSION}"
cp "$ROOT"/docs/images/*.png "$SITE/images/"

cd "$SITE/vendor/ort-${ORT_VERSION}"
cat > SHA256SUMS <<SUMS
219e6a1fc8a9938268d18efca3c91d310bd2f4a59bbd13744df5b2b7fc6cee3b  ort.wasm.min.mjs
e13f7f94fc51b4ca72b12faeb1ee95f4ace6dfbc8939bc718aabdc0a27c4299b  ort-wasm-simd-threaded.mjs
3398c10d07d229bd91b364548e130e0e51a8e5704b88c7c083ebbeb78842dee2  ort-wasm-simd-threaded.wasm
SUMS
for f in ort.wasm.min.mjs ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm; do
  [ -f "$f" ] || curl -sSfL --retry 3 -o "$f" "$ORT_URL/$f"
done
sha256sum -c SHA256SUMS
echo "site built in $SITE"
