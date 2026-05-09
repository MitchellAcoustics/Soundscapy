#!/bin/bash
# update-environment-yml.sh
# Usage: ./update-environment-yml.sh [environment.yml] [pyproject.toml]

ENV_FILE="${1:-environment.yml}"
PYPROJECT="${2:-pyproject.toml}"

# Extract version from pyproject.toml
VERSION=$(grep '^version = ' "$PYPROJECT" | sed 's/version = "\([^"]*\)"/\1/')

if [ -z "$VERSION" ]; then
    echo "Error: Could not extract version from $PYPROJECT" >&2
    exit 1
fi

echo "Found soundscapy version: $VERSION"

# Replace in environment.yml using sed
sed -i.bak \
    '/- pip:/{N;N;N;N;s|- pip:\n  - -e \.\[r\]\n  - -e \.\[audio\]\n  - -e \.|- pip:\n  - soundscapy[r,audio]>='"$VERSION"'|}' \
    "$ENV_FILE"

echo "✓ Updated $ENV_FILE with soundscapy version $VERSION"
