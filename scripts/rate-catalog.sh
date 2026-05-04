#!/bin/bash
# Rate genome catalog images using imv.
# Up arrow = good, Down arrow = bad, right = skip (leave in unsorted)
#
# Usage: scripts/rate-catalog.sh ~/genome_catalog

CATALOG="${1:?Usage: $0 <catalog_dir>}"
UNSORTED="$CATALOG/unsorted"
GOOD="$CATALOG/good"
BAD="$CATALOG/bad"

mkdir -p "$GOOD" "$BAD"

if [ -z "$(ls "$UNSORTED"/*.png 2>/dev/null)" ]; then
    echo "No PNGs in $UNSORTED"
    exit 1
fi

echo "Rating genomes in $UNSORTED"
echo "  Up    = good (move to good/)"
echo "  Down  = bad (move to bad/)"
echo "  Right = skip"
echo "  q     = quit"

imv -f \
    -c "bind <Up> exec mv \$imv_current_file $GOOD/ && imv-msg \$imv_pid close" \
    -c "bind <Down> exec mv \$imv_current_file $BAD/ && imv-msg \$imv_pid close" \
    -c "bind <Right> exec imv-msg \$imv_pid close" \
    "$UNSORTED"/*.png
