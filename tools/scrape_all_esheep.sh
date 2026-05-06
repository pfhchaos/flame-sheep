#!/bin/bash
# Scrape all Electric Sheep generations sequentially.
# Skips 247 if it's already been scraped.
# Run from the flame-sheep project root.

cd "$(dirname "$0")/.." || exit 1

for gen in 247 245 244 243 242 198 191 169 165; do
    count=$(sqlite3 ~/.local/share/flame-sheep/esheep.db \
        "SELECT COUNT(*) FROM sheep WHERE generation=$gen AND genome_xml IS NOT NULL;" 2>/dev/null)

    if [ "${count:-0}" -gt 100 ]; then
        echo "=== Generation $gen: already have $count genomes, skipping ==="
        continue
    fi

    echo "=== Scraping generation $gen ==="
    python tools/scrape_esheep.py --generation "$gen" --delay 2
    echo "=== Generation $gen done ==="
    echo ""
done

echo "=== All done ==="
sqlite3 ~/.local/share/flame-sheep/esheep.db \
    "SELECT generation, COUNT(*), SUM(genome_xml IS NOT NULL) as with_genome FROM sheep GROUP BY generation ORDER BY generation;"
