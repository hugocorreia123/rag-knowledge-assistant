#!/usr/bin/env bash
#
# Download the EU AI Act and related explanatory documents from EUR-Lex.
#
# Usage:
#   bash scripts/download_data.sh
#
# Files are saved to data/raw/. Re-running the script will skip files that
# already exist (idempotent).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

# Source documents — official EU AI Act in English from EUR-Lex.
#
# Regulation (EU) 2024/1689 — the EU AI Act
# Published 12 July 2024 in the Official Journal of the European Union.
declare -a SOURCES=(
    "eu_ai_act.pdf|https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"
)

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
download_if_missing() {
    local filename="$1"
    local url="$2"
    local target="$DATA_DIR/$filename"

    if [ -f "$target" ]; then
        echo "✓ Already present: $filename ($(du -h "$target" | cut -f1))"
        return 0
    fi

    echo "↓ Downloading $filename ..."
    curl --fail --location --silent --show-error \
        --user-agent "rag-knowledge-assistant/0.1" \
        --output "$target" \
        "$url"

    echo "  Saved: $target ($(du -h "$target" | cut -f1))"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "Downloading source documents to $DATA_DIR/ ..."
echo

for entry in "${SOURCES[@]}"; do
    filename="${entry%%|*}"
    url="${entry#*|}"
    download_if_missing "$filename" "$url"
done

echo
echo "Done. Files in $DATA_DIR/:"
ls -lh "$DATA_DIR/"