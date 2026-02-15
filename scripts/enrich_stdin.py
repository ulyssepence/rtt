#!/usr/bin/env python3
"""Enriches each stdin line via Claude immediately as it arrives.

Usage:
    cat segments.txt | uv run python scripts/enrich_stdin.py
    echo "a man walks into a factory" | uv run python scripts/enrich_stdin.py --context "1950s film"
"""

import argparse
import os
import sys

import anthropic

PROMPT = """You are an indexing assistant. Produce a short enriched version of the following transcript segment that adds related concepts, synonyms, and themes to make it more findable via semantic search. Preserve the original meaning. Output ONLY the enriched version, nothing else.

Context: {context}

Segment: {segment}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="general video content")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=512,
            messages=[{"role": "user", "content": PROMPT.format(context=args.context, segment=line)}],
        )
        print(resp.content[0].text.strip(), flush=True)


if __name__ == "__main__":
    main()
