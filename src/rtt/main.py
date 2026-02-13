import json
import shutil
from pathlib import Path

from rtt import types as t, transcribe, enrich, embed, frames, package


def _load_status(video_path: Path) -> dict:
    status_path = video_path.parent / f"{video_path.name}.rtt.json"
    if status_path.exists():
        return json.loads(status_path.read_text())
    return {"status": "new", "video_path": str(video_path)}


def _save_status(video_path: Path, status: dict):
    status_path = video_path.parent / f"{video_path.name}.rtt.json"
    status_path.write_text(json.dumps(status, indent=2))


def process(
    video_path: Path,
    video_id: str | None = None,
    title: str | None = None,
    source_url: str = "",
    context: str | None = None,
    skip_enrich: bool = False,
    output_dir: Path | None = None,
    collection: str = "",
) -> Path:
    vid_id = video_id or video_path.stem
    vid_title = title or video_path.stem
    vid_context = context or vid_title
    out = output_dir or video_path.parent

    status = _load_status(video_path)

    if status.get("status") == "ready":
        rtt_path = out / f"{vid_id}.rtt"
        if rtt_path.exists():
            return rtt_path

    transcriber = transcribe.WhisperTranscriber()
    enricher = None if skip_enrich else enrich.ClaudeEnricher()
    embedder = embed.OllamaEmbedder()

    if status.get("status") in ("new", "downloaded"):
        print(f"Transcribing {video_path}...")
        segments = transcriber.transcribe(video_path, vid_id)
        status["segments"] = [
            {"segment_id": s.segment_id, "start": s.start_seconds,
             "end": s.end_seconds, "text": s.transcript_raw}
            for s in segments
        ]
        status["status"] = "transcribed"
        _save_status(video_path, status)
    else:
        segments = [
            t.Segment(
                segment_id=s["segment_id"], video_id=vid_id,
                start_seconds=s["start"], end_seconds=s["end"],
                transcript_raw=s["text"],
            )
            for s in status["segments"]
        ]

    if skip_enrich:
        for seg in segments:
            seg.transcript_enriched = seg.transcript_raw
        status["status"] = "enriched"
    elif status.get("status") == "transcribed":
        print(f"Enriching {len(segments)} segments...")
        raw_texts = [s.transcript_raw for s in segments]
        enriched = enricher.enrich(vid_context, raw_texts)
        for seg, e in zip(segments, enriched):
            seg.transcript_enriched = e
        status["enriched"] = enriched
        status["status"] = "enriched"
        _save_status(video_path, status)
    elif "enriched" in status:
        for seg, e in zip(segments, status["enriched"]):
            seg.transcript_enriched = e

    if status.get("status") == "enriched":
        print(f"Embedding {len(segments)} segments...")
        texts = [s.transcript_enriched for s in segments]
        embeddings = embedder.embed_batch(texts)
        for seg, emb in zip(segments, embeddings):
            seg.text_embedding = emb
        status["status"] = "embedded"
        _save_status(video_path, status)
    else:
        texts = [s.transcript_enriched for s in segments]
        embeddings = embedder.embed_batch(texts)
        for seg, emb in zip(segments, embeddings):
            seg.text_embedding = emb

    print(f"Extracting frames...")
    frames_dir = video_path.parent / f"{video_path.name}.frames"
    timestamps = [s.start_seconds for s in segments]
    frame_paths = frames.extract(video_path, timestamps, frames_dir)
    for seg, fp in zip(segments, frame_paths):
        seg.frame_path = f"frames/{fp.name}" if fp else ""

    for seg in segments:
        seg.collection = collection
    duration = max(s.end_seconds for s in segments) if segments else 0.0
    video = t.Video(
        video_id=vid_id, title=vid_title, source_url=source_url,
        context=vid_context, duration_seconds=duration, status="ready",
        collection=collection,
    )

    out.mkdir(parents=True, exist_ok=True)
    rtt_path = out / f"{vid_id}.rtt"
    print(f"Packaging {rtt_path}...")
    package.create(video, segments, frames_dir, rtt_path)

    status_path = video_path.parent / f"{video_path.name}.rtt.json"
    status_path.unlink(missing_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    return rtt_path
