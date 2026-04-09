#!/usr/bin/env bash
# Quick analysis script for OpenClaw integration
# Usage: ./scripts/analyze.sh BHP.AX,CBA.AX [json|table]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

TICKERS="${1:?Usage: analyze.sh TICKERS [json|table]}"
OUTPUT="${2:-json}"

cd "$PROJECT_DIR"
python3 -m src.main --tickers "$TICKERS" --output "$OUTPUT"
