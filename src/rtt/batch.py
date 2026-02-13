import asyncio
import json
import shutil
from pathlib import Path

from rtt import types as t, transcribe, enrich, embed, frames, package, youtube

DOWNLOAD_CONCURRENCY = 5
TRANSCRIBE_CONCURRENCY = 5
ENRICH_CONCURRENCY = 5


def _status_path(output_dir: Path, video_id: str) -> Path:
    return output_dir / f"{video_id}.rtt.json"


def _frames_dir(output_dir: Path, video_id: str) -> Path:
    return output_dir / f"{video_id}.frames"


def _load_status(output_dir: Path, video_id: str) -> dict:
    p = _status_path(output_dir, video_id)
    if p.exists():
        return json.loads(p.read_text())
    return {"status": "new"}


def _save_status(output_dir: Path, video_id: str, status: dict):
    _status_path(output_dir, video_id).write_text(json.dumps(status, indent=2))


def _cleanup(output_dir: Path, video_id: str):
    _status_path(output_dir, video_id).unlink(missing_ok=True)
    fd = _frames_dir(output_dir, video_id)
    if fd.exists():
        shutil.rmtree(fd)


async def process_batch(
    jobs: list[t.VideoJob],
    output_dir: Path,
    skip_enrich: bool = False,
    failures_path: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fail_path = failures_path or output_dir / "failures.jsonl"

    transcriber = transcribe.AssemblyAITranscriber()
    enricher = None if skip_enrich else enrich.ClaudeEnricher()
    embedder = embed.OllamaEmbedder()

    sem_download = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    sem_transcribe = asyncio.Semaphore(TRANSCRIBE_CONCURRENCY)
    sem_enrich = asyncio.Semaphore(ENRICH_CONCURRENCY)
    fail_lock = asyncio.Lock()

    async def _log_failure(job: t.VideoJob, error: str):
        entry = json.dumps({"video_id": job.video_id, "source_url": job.source_url,
                            "title": job.title, "error": error})
        async with fail_lock:
            with open(fail_path, "a") as f:
                f.write(entry + "\n")

    async def _process_one(job: t.VideoJob) -> Path | None:
        vid = job.video_id
        rtt_path = output_dir / f"{vid}.rtt"
        if rtt_path.exists():
            print(f"[{vid}] Already packaged, skipping")
            return rtt_path

        audio_path = output_dir / f"{vid}.audio"
        video_path = output_dir / f"{vid}.video"

        try:
            status = _load_status(output_dir, vid)

            # --- Download audio ---
            if status.get("status") == "new":
                print(f"[{vid}] Downloading audio...")
                async with sem_download:
                    dl_path = await asyncio.to_thread(
                        youtube.RealDownloader.download_audio, vid, output_dir,
                    )
                    dl_path.rename(audio_path)
                status["status"] = "downloaded"
                _save_status(output_dir, vid, status)
                print(f"[{vid}] Downloaded audio")

            # --- Transcribe ---
            if status.get("status") == "downloaded":
                print(f"[{vid}] Transcribing...")
                async with sem_transcribe:
                    segments = await asyncio.to_thread(
                        transcriber.transcribe_url, str(audio_path), vid,
                    )
                audio_path.unlink(missing_ok=True)
                if not segments:
                    await _log_failure(job, "No segments returned")
                    print(f"[{vid}] FAILED: no segments")
                    return None
                status["segments"] = [
                    {"segment_id": s.segment_id, "start": s.start_seconds,
                     "end": s.end_seconds, "text": s.transcript_raw}
                    for s in segments
                ]
                status["status"] = "transcribed"
                _save_status(output_dir, vid, status)
                print(f"[{vid}] Transcribed ({len(segments)} segments)")
            else:
                audio_path.unlink(missing_ok=True)
                segments = [
                    t.Segment(segment_id=s["segment_id"], video_id=vid,
                              start_seconds=s["start"], end_seconds=s["end"],
                              transcript_raw=s["text"])
                    for s in status["segments"]
                ]
                if status.get("status") != "new":
                    print(f"[{vid}] Resuming from {status['status']}")

            # --- Enrich ---
            if skip_enrich:
                for seg in segments:
                    seg.transcript_enriched = seg.transcript_raw
            elif status.get("status") == "transcribed":
                print(f"[{vid}] Enriching...")
                async with sem_enrich:
                    raw_texts = [s.transcript_raw for s in segments]
                    enriched = await asyncio.to_thread(
                        enricher.enrich, job.context or job.title, raw_texts,
                    )
                    for seg, e in zip(segments, enriched):
                        seg.transcript_enriched = e
                status["enriched"] = [s.transcript_enriched for s in segments]
                status["status"] = "enriched"
                _save_status(output_dir, vid, status)
                print(f"[{vid}] Enriched")
            else:
                for seg, e in zip(segments, status.get("enriched", [])):
                    seg.transcript_enriched = e

            # --- Embed ---
            if status.get("status") in ("enriched", "transcribed"):
                print(f"[{vid}] Embedding...")
                texts = [s.transcript_enriched for s in segments]
                embeddings = await asyncio.to_thread(embedder.embed_batch, texts)
                for seg, emb in zip(segments, embeddings):
                    seg.text_embedding = emb
                status["status"] = "embedded"
                _save_status(output_dir, vid, status)
                print(f"[{vid}] Embedded")
            else:
                texts = [s.transcript_enriched for s in segments]
                embeddings = await asyncio.to_thread(embedder.embed_batch, texts)
                for seg, emb in zip(segments, embeddings):
                    seg.text_embedding = emb

            # --- Frames ---
            print(f"[{vid}] Downloading video for frames...")
            async with sem_download:
                dl_path = await asyncio.to_thread(
                    youtube.RealDownloader.download_video, vid, output_dir,
                )
                dl_path.rename(video_path)
            fd = _frames_dir(output_dir, vid)
            fd.mkdir(exist_ok=True)
            timestamps = [s.start_seconds for s in segments]
            frame_paths = await asyncio.to_thread(
                frames.extract, video_path, timestamps, fd,
            )
            video_path.unlink(missing_ok=True)
            for seg, fp in zip(segments, frame_paths):
                seg.frame_path = f"frames/{fp.name}" if fp else ""

            # --- Package ---
            duration = max(s.end_seconds for s in segments) if segments else 0.0
            video = t.Video(
                video_id=vid, title=job.title,
                source_url=job.page_url or job.source_url,
                context=job.context or job.title,
                duration_seconds=duration, status="ready",
            )
            package.create(video, segments, fd, rtt_path)
            _cleanup(output_dir, vid)
            video_path.unlink(missing_ok=True)
            print(f"[{vid}] Done -> {rtt_path}")
            return rtt_path

        except Exception as exc:
            await _log_failure(job, f"{type(exc).__name__}: {exc}")
            audio_path.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)
            print(f"[{vid}] FAILED: {exc}")
            return None

    tasks = [_process_one(job) for job in jobs]
    results = await asyncio.gather(*tasks)
    paths = [p for p in results if p is not None]
    n_failed = len(jobs) - len(paths)
    print(f"\nBatch complete: {len(paths)}/{len(jobs)} succeeded")
    if n_failed > 0:
        print(f"Failures logged to {fail_path}")
    return paths
