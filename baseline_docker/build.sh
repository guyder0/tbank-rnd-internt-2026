#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$package_dir/prepare_context.py"
docker build --tag "${1:-kion-baseline:latest}" "$package_dir/.build-context"
