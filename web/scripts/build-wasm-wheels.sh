#!/bin/bash
# Cross-compiles the Nixtla forecasting stack (coreforecast, statsforecast)
# into wasm32/emscripten wheels for the exact Pyodide this app ships, because
# PyPI carries only native builds for them. Output lands in
# vendor/pyodide-wasm-wheels/, which scripts/setup-runtimes.mjs copies into
# the served wheel set (with an ABI check, so a stale build fails loudly).
#
# Run from webapp/web with Docker available:
#   bash scripts/build-wasm-wheels.sh
#
# Re-run after any Pyodide upgrade (the wheels are ABI-pinned) or to pick up
# new statsforecast releases (bump the versions below).
set -euo pipefail

COREFORECAST_VERSION="0.0.18"
STATSFORECAST_VERSION="2.1.1"

here="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

sdist_url() {
  curl -s "https://pypi.org/pypi/$1/$2/json" |
    python -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['packagetype']=='sdist'][0])"
}

echo "Fetching sdists..."
curl -sL -o "$work/coreforecast.tar.gz" "$(sdist_url coreforecast "$COREFORECAST_VERSION")"
curl -sL -o "$work/statsforecast.tar.gz" "$(sdist_url statsforecast "$STATSFORECAST_VERSION")"

cat > "$work/inner.sh" <<INNER
set -euxo pipefail
apt-get update -qq
apt-get install -y -qq git cmake ninja-build build-essential > /dev/null
pip install --quiet pyodide-build
EMSCRIPTEN_VERSION=\$(pyodide config get emscripten_version)
git clone --quiet --depth 1 https://github.com/emscripten-core/emsdk /emsdk
cd /emsdk && ./emsdk install "\$EMSCRIPTEN_VERSION" && ./emsdk activate "\$EMSCRIPTEN_VERSION"
source /emsdk/emsdk_env.sh
mkdir -p /build && cd /build
tar -xzf /work/coreforecast.tar.gz
tar -xzf /work/statsforecast.tar.gz
cd /build/coreforecast-${COREFORECAST_VERSION}
# coreforecast sets PYBIND11_NEWPYTHON, which is not a pybind11 variable, so
# the legacy FindPythonLibsNew path runs and fails its 64-vs-32-bit check
# under cross-compilation. Forcing pybind11's real modern toggle fixes it.
CMAKE_ARGS="-DPYBIND11_FINDPYTHON=ON" pyodide build
cp dist/*.whl /out/
cd /build/statsforecast-${STATSFORECAST_VERSION}
pyodide build
cp dist/*.whl /out/
ls -la /out
echo "BUILD OK"
INNER

mkdir -p "$here/vendor/pyodide-wasm-wheels"
docker run --rm \
  -v "$work:/work" \
  -v "$here/vendor/pyodide-wasm-wheels:/out" \
  python:3.14 bash /work/inner.sh

echo "Wheels in vendor/pyodide-wasm-wheels; now run: npm run setup:runtimes -- pypi-wheels"
