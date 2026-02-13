# Remember That Time (RTT)

Semantic video search command line tool and web demo.

The command line tool transcribes videos, enriches transcripts with an LLM for better retrieval ([EnrichIndex](https://arxiv.org/html/2504.03598)), embeds everything into a vector database, and finally writes it to a zip file with a specific structure (an `.rtt` file).

The web server takes a directory of `rtt` files that have been . Search in natural language, get frozen frames from matching moments. Click a frame, play the video from that timestamp.

Built for the [Prelinger Archives](https://archive.org/details/prelinger) — thousands of mid-century educational, industrial, and propaganda films, all public domain on Internet Archive.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [FFmpeg](https://ffmpeg.org/)
- [Ollama](https://ollama.com/) with `nomic-embed-text` pulled (`ollama pull nomic-embed-text`)
- Anthropic API key (for transcript enrichment)
- AssemblyAI API key (for batch/YouTube processing)

## Setup

```
uv sync
cp .env.example .env        # add your ANTHROPIC_API_KEY
```

## Usage

Process a video (transcribe → enrich → embed → extract frames → package):

```
uv run rtt process video.mp4
```

Process a directory of videos:

```
uv run rtt process data/videos/
```

Serve the search UI over processed `.rtt` files:

```
uv run rtt serve data/videos/
```

Then open `http://localhost:8000`.

Batch process an entire YouTube channel:

```
uv run rtt batch "https://www.youtube.com/@channel" -o output/
```

Batch process from a JSON manifest:

```
uv run rtt batch jobs.json -o output/
```

### Individual pipeline stages

```
uv run rtt transcribe video.mp4
uv run rtt enrich video.mp4
uv run rtt embed video.mp4
```

## How it works

1. **Transcribe** — `faster-whisper` (large-v3, local) for single files, or AssemblyAI (cloud) for batch/YouTube. Produces timestamped transcript segments
2. **Enrich** — Claude rewrites each segment to add related concepts, synonyms, and themes (EnrichIndex technique, +11.7 recall@10 over raw transcripts)
3. **Embed** — `nomic-embed-text` via Ollama produces 768d vectors for each enriched segment
4. **Extract frames** — FFmpeg pulls a thumbnail at each segment's start timestamp
5. **Package** — everything bundles into a portable `.rtt` file (zip containing `manifest.json`, `segments.parquet`, and `frames/`)
6. **Serve** — FastAPI loads `.rtt` files into an in-memory LanceDB, serves search + thumbnails + a React frontend with Plyr.js video player

## The `.rtt` format

```
video.rtt (zip)
├── manifest.json        # video metadata + segment data
├── segments.parquet     # all segments + 768d embeddings (Arrow columnar)
└── frames/
    000012.jpg
    000045.jpg
```

## Tests

```
uv run pytest                           # all tests
uv run pytest -k "not e2e"              # skip slow end-to-end
uv run pytest tests/test_server.py -v   # just server tests
```

## Architecture

All modules in `src/rtt/`, flat structure. Protocol classes for interfaces, one implementation each. No DI framework — everything wired in `main.py`.

| Module | Responsibility |
|---|---|
| `transcribe.py` | Whisper (local) and AssemblyAI (cloud) transcription |
| `youtube.py` | YouTube channel listing and video download (yt-dlp) |
| `batch.py` | Async batch pipeline with resumable status tracking |
| `enrich.py` | Claude transcript enrichment |
| `embed.py` | Ollama text embeddings |
| `frames.py` | FFmpeg frame extraction |
| `vector.py` | LanceDB vector search |
| `package.py` | `.rtt` file creation/loading |
| `server.py` | FastAPI search API + frontend |
| `main.py` | Pipeline orchestration |

## Cost

Whisper and embeddings run locally (free). Claude enrichment: ~$0.03 per video for a 10-minute film with Sonnet. AssemblyAI transcription (batch mode): ~$0.06 per video for a 10-minute film.
