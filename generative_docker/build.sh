#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$package_dir/../baseline/.venv/bin/python" "$package_dir/prepare_context.py"
docker build --tag "${1:-kion-generative:latest}" "$package_dir/.build-context"
