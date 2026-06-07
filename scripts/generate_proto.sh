#!/usr/bin/env bash
# Generate Python gRPC stubs from proto/inference.proto into generated/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$ROOT/generated"
touch "$ROOT/generated/__init__.py"

python3 -m grpc_tools.protoc \
  --proto_path="$ROOT/proto" \
  --python_out="$ROOT/generated" \
  --grpc_python_out="$ROOT/generated" \
  "$ROOT/proto/inference.proto"

echo "Proto stubs written to $ROOT/generated/"
