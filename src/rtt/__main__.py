import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import dotenv
dotenv.load_dotenv()

VIDEO_EXTS = (".mp4", ".webm", ".mkv")


def _is_url(s: str) -> bool:
    return urlparse(s).scheme in ("http", "https")


def _download(url: str, dest_dir: Path) -> Path:
    import httpx
    filename = Path(urlparse(url).path).name or "video.mp4"
    dest = dest_dir / filename
    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return dest
    print(f"  Downloading {url}...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                f.write(chunk)
    print(f"  Saved to {dest}")
    return dest


def _resolve_local(raw_args: list[str]) -> tuple[list[tuple[Path, str]], list[str]]:
    local: list[tuple[Path, str]] = []
    urls: list[str] = []
    for arg in raw_args:
        if _is_url(arg):
            urls.append(arg)
        else:
            p = Path(arg)
            if p.is_file():
                local.append((p, ""))
            elif p.is_dir():
                for ext in VIDEO_EXTS:
                    for f in sorted(p.glob(f"*{ext}")):
                        local.append((f, ""))
            else:
                print(f"Warning: skipping {arg} (not a file, directory, or URL)")
    return local, urls


def main():
    parser = argparse.ArgumentParser(prog="rtt")
    sub = parser.add_subparsers(dest="command")

    p_process = sub.add_parser("process")
    p_process.add_argument("paths", nargs="+")
    p_process.add_argument("--output-dir", "-o", type=Path, default=Path("."), help="Directory for downloaded videos and their .rtt files")
    p_process.add_argument("--title", type=str, default=None)
    p_process.add_argument("--context", type=str, default=None)
    p_process.add_argument("--no-enrich", action="store_true", help="Skip LLM enrichment (no API key needed)")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("directory", type=Path)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    p_transcribe = sub.add_parser("transcribe")
    p_transcribe.add_argument("path", type=Path)

    p_enrich = sub.add_parser("enrich")
    p_enrich.add_argument("path", type=Path)

    p_embed = sub.add_parser("embed")
    p_embed.add_argument("path", type=Path)

    p_channel = sub.add_parser("channel", help="List video IDs from a YouTube channel")
    p_channel.add_argument("url", type=str)

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("input", type=str, help="JSON file, directory of JSON files, or YouTube channel URL")
    p_batch.add_argument("--output-dir", "-o", type=Path, default=Path("."))
    p_batch.add_argument("--no-enrich", action="store_true")

    args = parser.parse_args()

    from rtt import runtime

    if args.command == "process":
        runtime.require(needs_ffmpeg=True, needs_ollama=True, needs_anthropic=not args.no_enrich)
        from rtt import main as pipeline
        import tempfile
        local, urls = _resolve_local(args.paths)
        if not local and not urls:
            print("No video files found.")
            sys.exit(1)
        for video_path, source_url in local:
            pipeline.process(video_path, title=args.title, source_url=source_url,
                             context=args.context, skip_enrich=args.no_enrich)
        for url in urls:
            with tempfile.TemporaryDirectory(prefix="rtt_dl_") as tmp:
                video_path = _download(url, Path(tmp))
                pipeline.process(video_path, title=args.title, source_url=url,
                                 context=args.context, skip_enrich=args.no_enrich,
                                 output_dir=args.output_dir)

    elif args.command == "serve":
        runtime.require(needs_ollama=True)
        import uvicorn
        from rtt import server
        app = server.create_app(args.directory)
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "transcribe":
        runtime.require(needs_ffmpeg=True)
        from rtt import transcribe as tr
        t = tr.WhisperTranscriber()
        segs = t.transcribe(args.path, args.path.stem)
        for s in segs:
            print(f"[{s.start_seconds:.1f}-{s.end_seconds:.1f}] {s.transcript_raw}")

    elif args.command == "enrich":
        runtime.require(needs_anthropic=True)
        import json
        from rtt import enrich as en
        status_path = args.path.parent / f"{args.path.name}.rtt.json"
        status = json.loads(status_path.read_text())
        texts = [s["text"] for s in status["segments"]]
        enricher = en.ClaudeEnricher()
        enriched = enricher.enrich(args.path.stem, texts)
        for r, e in zip(texts, enriched):
            print(f"RAW: {r}\nENRICHED: {e}\n")

    elif args.command == "embed":
        runtime.require(needs_ollama=True)
        from rtt import embed as em
        import json
        status_path = args.path.parent / f"{args.path.name}.rtt.json"
        status = json.loads(status_path.read_text())
        texts = status.get("enriched", [s["text"] for s in status["segments"]])
        embedder = em.OllamaEmbedder()
        vecs = embedder.embed_batch(texts)
        print(f"Embedded {len(vecs)} segments, dim={len(vecs[0])}")

    elif args.command == "batch":
        runtime.require(
            needs_ffmpeg=True, needs_ollama=True,
            needs_assemblyai=True, needs_anthropic=not args.no_enrich,
        )
        import asyncio, json
        from rtt import batch, types as t_mod
        if _is_url(args.input) and "youtube.com/" in args.input:
            from rtt import youtube
            entries = youtube.channel_video_ids(args.input)
            print(f"Found {len(entries)} videos")
            jobs = [
                t_mod.VideoJob(
                    video_id=e["id"],
                    title=e["title"],
                    source_url=youtube.video_url(e["id"]),
                    page_url=youtube.video_url(e["id"]),
                )
                for e in entries
            ]
        else:
            input_path = Path(args.input)
            if input_path.is_dir():
                raw = []
                for f in sorted(input_path.glob("*.json")):
                    raw.extend(json.loads(f.read_text()) if isinstance(json.loads(f.read_text()), list) else [json.loads(f.read_text())])
            else:
                data = json.loads(input_path.read_text())
                raw = data if isinstance(data, list) else [data]
            jobs = [t_mod.VideoJob(**j) for j in raw]
        if not jobs:
            print("No video jobs found.")
            sys.exit(1)
        print(f"Processing {len(jobs)} videos in batch mode...")
        asyncio.run(batch.process_batch(jobs, args.output_dir, skip_enrich=args.no_enrich))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
