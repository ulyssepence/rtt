# AGENTS.md

## What this is

RTT (Remember That Time) — semantic video search over the Prelinger Archives. Ingestion pipeline turns public domain films into searchable `.rtt` files; FastAPI serves search over them; React frontend shows results with inline playback.

See `docs/2026-02-11_RTT semantic video search PRD.md` for full spec.

## Architecture

```
CLI (__main__.py)
 └─ main.process()
     ├─ transcribe.WhisperTranscriber  (faster-whisper, local)
     ├─ enrich.ClaudeEnricher          (Anthropic API)
     ├─ embed.OllamaEmbedder           (Ollama nomic-embed-text, local)
     ├─ frames.extract()               (FFmpeg)
     └─ package.create()               (→ .rtt zip)

CLI (__main__.py)
 └─ server.create_app(rtt_dir)
     ├─ package.load()     (reads .rtt files)
     ├─ vector.Database    (LanceDB in-memory)
     └─ FastAPI routes     (GET /search, /video/{id}, static frames, frontend)
```

## Module map

All code in `src/rtt/`, flat structure. Types in `types.py`. Qualified imports: `from rtt import transcribe; transcribe.WhisperTranscriber`.

| Module | Key types | External deps |
|--------|-----------|---------------|
| `types.py` | `Segment`, `Video` dataclasses | — |
| `transcribe.py` | `Transcriber` protocol, `WhisperTranscriber` | faster-whisper (local) |
| `enrich.py` | `Enricher` protocol, `ClaudeEnricher` | Anthropic API |
| `embed.py` | `Embedder` protocol, `OllamaEmbedder` | Ollama HTTP (localhost:11434) |
| `frames.py` | `extract()` | FFmpeg (system binary) |
| `vector.py` | `Database` (wraps LanceDB) | lancedb, pyarrow |
| `package.py` | `create()`, `load()` | pyarrow, zipfile |
| `server.py` | `create_app()` → FastAPI | fastapi, uvicorn |
| `main.py` | `process()` — pipeline orchestrator | wires all above |
| `runtime.py` | `check()`, `require()`, `ensure_whisper()` | — |
| `__main__.py` | CLI: `process`, `serve`, `transcribe`, `enrich`, `embed` | argparse |

## The `.rtt` format

Zip archive containing `manifest.json` + `segments.parquet` + `frames/*.jpg`. The portable artifact. Server loads these at boot, merges into single LanceDB table.

## Pipeline status tracking

`main.process()` writes a `.rtt.json` sidecar during processing: `new → transcribed → enriched → embedded → ready`. Pipeline is resumable — skips completed stages. On completion, packages into `.rtt` and cleans up intermediates.

## Running

```bash
# Prerequisites: FFmpeg installed, Ollama running with nomic-embed-text, ANTHROPIC_API_KEY set
uv run rtt process video.mp4                    # local file → .rtt next to it
uv run rtt process data/sample/                 # directory → all videos in it
uv run rtt process https://archive.org/download/Film/film.mp4  # URL → downloads, processes, .rtt in cwd
uv run rtt process a.mp4 dir/ https://url/v.mp4 # mix files, dirs, URLs
uv run rtt process video.mp4 --no-enrich        # skip Claude enrichment
uv run rtt process url -o out/                  # override output dir for URL .rtt files
uv run rtt serve data/sample/                   # start server
uv run rtt transcribe video.mp4                 # single stage
```

## Tests

```bash
uv run pytest                                    # all tests
uv run pytest tests/test_embed.py               # unit: Ollama embeddings
uv run pytest tests/test_vector.py              # unit: LanceDB ops
uv run pytest tests/test_server.py              # integration: FastAPI
uv run pytest tests/test_e2e.py                 # e2e: full pipeline + search quality
```

Tests requiring external services (Whisper, Ollama, Anthropic) hit real services — no mocks except `FakeEmbedder` in server tests for deterministic search assertions. Sample video in `data/sample/`.

## Conventions

- **Protocols over ABCs** — `Transcriber`, `Enricher`, `Embedder` are `typing.Protocol`
- **Plain dataclasses** for data types, no ORMs
- **No comments** unless something is genuinely surprising
- **No DI framework** — `main.py` instantiates and wires implementations directly
- **Flat file structure** — no nested packages
- **Qualified imports** — `from rtt import embed; embed.OllamaEmbedder`

## Frontend

`frontend/src/app.ts` compiled to `frontend/dist/app.js` via esbuild. Dark theme, search grid, Plyr.js video player with timestamp seeking. Served as static files by FastAPI.
