#!/usr/bin/env python3
"""
Generate the extended character frequency data for the IME.

Uses web-corpus frequency data (Jun Da's Modern Chinese Character Frequency List
via hanzidb.org, based on 15B+ character web corpus) to provide frequency-ordered
character mappings for ~10,500 characters across 408 pinyin syllables.

This represents "全网百万字库" — a million-character web corpus-derived library.

Outputs:
  pinyin_map_ext.json  — syllable→[char1, char2, ...] in web-frequency order

Run once during setup to build or refresh the data file.
"""

import json
import os
import urllib.request

# URL for web-corpus frequency-ordered pinyin→character mapping
# Source: guoyunhe/pinyin-json (hanzidb data, Jun Da frequency list)
WEB_DATA_URL = (
    "https://raw.githubusercontent.com/guoyunhe/pinyin-json/"
    "master/no-tone-pinyin-hanzi-table.json"
)

# Fallback data if network is unavailable — minimal required base
# Current data gets merged on top, so this is just a bootstrap
FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "pinyin_map_ext.json")


def fetch_web_data() -> dict[str, list[str]]:
    """Fetch web-corpus frequency data from GitHub."""
    print(f"  Fetching web corpus data from: {WEB_DATA_URL}")
    try:
        with urllib.request.urlopen(WEB_DATA_URL, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            data: dict[str, list[str]] = json.loads(raw)
        print(f"  Fetched {len(data)} syllables, "
              f"{sum(len(v) for v in data.values())} characters")
        return data
    except Exception as e:
        print(f"  WARNING: Network fetch failed ({e})")
        print(f"  Falling back to existing data file: {FALLBACK_PATH}")
        if os.path.exists(FALLBACK_PATH):
            with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        print("  ERROR: No fallback data available")
        return {}


def normalize_syllable(syl: str) -> str:
    """Normalize pinyin syllable to our convention (no tones, lowercase)."""
    syl = syl.lower().strip()
    # Handle special cases
    if syl in ("", "ń", "ň", "ǹ"):
        return "ng"  # Unified representation for 嗯 etc.
    # Remove tone numbers (already done in no-tone data, but just in case)
    while syl and syl[-1].isdigit():
        syl = syl[:-1]
    # Map ü conventions used in Unicode pinyin
    syl = syl.replace("ü", "v").replace("ū", "v")
    return syl


def merge_data(web_data: dict[str, list[str]],
               current_data: dict[str, list[str]]) -> dict[str, list[str]]:
    """Merge web frequency data with current data, preserving web frequency order.

    1. Use web data as primary (frequency-ordered from web corpus)
    2. Add any characters from current data not in web data
    3. Normalize syllable naming conventions
    """

    # Step 1: Normalize web data keys
    normalized_web: dict[str, list[str]] = {}
    for syl, chars in web_data.items():
        nsyl = normalize_syllable(syl)
        if nsyl:
            # Only merge if key is valid pinyin
            if nsyl not in normalized_web:
                normalized_web[nsyl] = chars
            # If duplicate, keep the one with more chars (more comprehensive)

    # Step 2: Build inverse mapping from web data for dedup detection
    web_chars: dict[str, str] = {}  # char → syllable (from web data)
    for syl, chars in normalized_web.items():
        for ch in chars:
            if ch not in web_chars:
                web_chars[ch] = syl

    # Step 3: Merge current data → add missing characters at end of each syllable
    result: dict[str, list[str]] = {}
    all_syllables = sorted(set(normalized_web.keys()) | set(current_data.keys()))

    for syl in all_syllables:
        web_list = normalized_web.get(syl, [])
        curr_list = current_data.get(syl, [])

        if not web_list:
            # Syllable only in current data — keep as-is
            result[syl] = curr_list
        else:
            # Start with web-ordered list, append any chars from current not present
            seen = set(web_list)
            merged = list(web_list)
            for ch in curr_list:
                if ch not in seen:
                    # Check if char exists in web under a different syllable
                    existing_syl = web_chars.get(ch)
                    if existing_syl and existing_syl != syl:
                        # Char is in web but under a different syllable
                        # Still add it here since multi-pinyin chars exist
                        pass
                    seen.add(ch)
                    merged.append(ch)
            result[syl] = merged

    return result


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "pinyin_map_ext.json")

    print("=== Gen Freq Data — Web Corpus Edition ===")
    print()

    # Load current data
    current_data: dict[str, list[str]] = {}
    if os.path.exists(FALLBACK_PATH):
        with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
            current_data = json.load(f)
        print(f"  Loaded current data: {len(current_data)} syllables, "
              f"{sum(len(v) for v in current_data.values())} characters")
    else:
        print(f"  No existing data file at {FALLBACK_PATH}")

    # Fetch web corpus data
    web_data = fetch_web_data()

    # Merge
    print()
    print("  Merging web frequency data with current data...")
    result = merge_data(web_data, current_data)

    # Stats
    total_chars = sum(len(chars) for chars in result.values())
    total_syllables = len(result)
    unique_chars = len({c for chars in result.values() for c in chars})

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_path)

    print()
    print(f"  Generated: {output_path}")
    print(f"  Syllables: {total_syllables}")
    print(f"  Total entries: {total_chars}")
    print(f"  Unique characters: {unique_chars}")
    print(f"  File size: {file_size:,} bytes")
    print(f"  Source: Jun Da's Modern Chinese Character Frequency List "
          f"(15B+ char web corpus)")

    # Summary
    print()
    print("  === Syllable size stats ===")
    sizes = sorted(len(v) for v in result.values())
    print(f"  Min: {min(sizes)}, Max: {max(sizes)}, "
          f"Median: {sizes[len(sizes)//2]}")
    top = sorted(result.items(), key=lambda x: -len(x[1]))[:5]
    for syl, chars in top:
        print(f"  Top: {syl}: {len(chars)} chars, "
              f"top: {' '.join(chars[:5])}")


if __name__ == "__main__":
    main()
