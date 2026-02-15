#!/usr/bin/env python3
"""Embeds each stdin line via Ollama and prints the embedding as a JSON array.

Usage:
    echo "a man in a factory" | uv run python scripts/embed_stdin.py
    cat enriched.txt | uv run python scripts/embed_stdin.py --ollama-url http://localhost:11434
"""

import argparse
import json
import sys

import httpx

OLLAMA_URL = "http://localhost:11434"
MODEL = "nomic-embed-text"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    client = httpx.Client(timeout=60)

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        resp = client.post(
            f"{args.ollama_url}/api/embed",
            json={"model": args.model, "input": [line]},
        )
        resp.raise_for_status()
        print(json.dumps(resp.json()["embeddings"][0]), flush=True)


if __name__ == "__main__":
    main()
