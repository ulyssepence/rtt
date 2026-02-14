import asyncio
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from rtt import types as t, transcribe, enrich, embed, frames, package, youtube, normalize

DEFAULT_TRANSCRIBE_CONCURRENCY = 20
DEFAULT_ENRICH_CONCURRENCY = 10
DEFAULT_EMBED_CONCURRENCY = 3
DEFAULT_FRAMES_CONCURRENCY = 3


def _download_url(url: str, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
    return path


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


@dataclass
class _Job:
    job: t.VideoJob
    status: dict = field(default_factory=dict)
    segments: list[t.Segment] = field(default_factory=list)
    error: str | None = None
    queued_at: dict[str, float] = field(default_factory=dict)


async def process_batch(
    jobs: list[t.VideoJob],
    output_dir: Path,
    skip_enrich: bool = False,
    failures_path: Path | None = None,
    concurrency_transcribe: int = DEFAULT_TRANSCRIBE_CONCURRENCY,
    concurrency_enrich: int = DEFAULT_ENRICH_CONCURRENCY,
    concurrency_embed: int = DEFAULT_EMBED_CONCURRENCY,
    concurrency_frames: int = DEFAULT_FRAMES_CONCURRENCY,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fail_path = failures_path or output_dir / "failures.jsonl"

    yt_transcriber = transcribe.YouTubeTranscriber()
    aai_transcriber = transcribe.AssemblyAITranscriber()
    enricher = None if skip_enrich else enrich.ClaudeEnricher()
    embedder = embed.OllamaEmbedder()

    q_transcribe: asyncio.Queue[_Job] = asyncio.Queue()
    q_enrich: asyncio.Queue[_Job] = asyncio.Queue()
    q_embed: asyncio.Queue[_Job] = asyncio.Queue()
    q_frames: asyncio.Queue[_Job] = asyncio.Queue()

    results: list[Path] = []
    fail_lock = asyncio.Lock()
    done_count = 0
    total = len(jobs)

    async def _log_failure(job: t.VideoJob, error: str):
        entry = json.dumps({"video_id": job.video_id, "source_url": job.source_url,
                            "title": job.title, "error": error})
        async with fail_lock:
            with open(fail_path, "a") as f:
                f.write(entry + "\n")

    def _is_youtube(j: _Job) -> bool:
        return "youtube.com" in j.job.source_url or "youtu.be" in j.job.source_url

    # --- Stage: Transcribe ---
    async def transcribe_worker():
        while True:
            j = await q_transcribe.get()
            vid = j.job.video_id
            waited = time.monotonic() - j.queued_at.get("transcribe", time.monotonic())
            try:
                segments = None
                if _is_youtube(j):
                    print(f"[{vid}] Trying YouTube subtitles...")
                    segments = await asyncio.to_thread(yt_transcriber.transcribe, vid)
                    if segments:
                        segments = normalize.normalize(segments)
                        j.status["transcript_source"] = "youtube"
                        print(f"[{vid}] Got YouTube subtitles ({len(segments)} segments)")

                if not segments:
                    audio_path = output_dir / f"{vid}.audio"
                    if _is_youtube(j):
                        print(f"[{vid}] No subtitles, falling back to AssemblyAI")
                        print(f"[{vid}] Downloading audio...")
                        dl_path = await asyncio.to_thread(
                            youtube.RealDownloader.download_audio, vid, output_dir,
                        )
                        dl_path.rename(audio_path)
                        transcribe_source = str(audio_path)
                    else:
                        transcribe_source = j.job.source_url
                    t0 = time.monotonic()
                    print(f"[{vid}] Transcribing...")
                    segments = await asyncio.to_thread(
                        aai_transcriber.transcribe_url, transcribe_source, vid,
                    )
                    print(f"[{vid}] Transcribed in {time.monotonic() - t0:.0f}s (waited {waited:.0f}s)")
                    audio_path.unlink(missing_ok=True)
                    if not segments:
                        await _log_failure(j.job, "No segments returned")
                        print(f"[{vid}] FAILED: no segments")
                        q_transcribe.task_done()
                        continue
                    segments = normalize.normalize(segments)
                    j.status["transcript_source"] = "assemblyai"

                if not segments:
                    await _log_failure(j.job, "No segments after normalization")
                    print(f"[{vid}] FAILED: no segments")
                    q_transcribe.task_done()
                    continue

                j.segments = segments
                j.status["segments"] = [
                    {"segment_id": s.segment_id, "start": s.start_seconds,
                     "end": s.end_seconds, "text": s.transcript_raw}
                    for s in segments
                ]
                j.status["status"] = "transcribed"
                _save_status(output_dir, vid, j.status)
                print(f"[{vid}] Transcribed ({len(segments)} segments)")
                j.queued_at["enrich"] = time.monotonic()
                q_enrich.put_nowait(j)
            except Exception as exc:
                await _log_failure(j.job, f"{type(exc).__name__}: {exc}")
                print(f"[{vid}] FAILED: {exc}")
            q_transcribe.task_done()

    # --- Stage: Enrich ---
    async def enrich_worker():
        while True:
            j = await q_enrich.get()
            vid = j.job.video_id
            waited = time.monotonic() - j.queued_at.get("enrich", time.monotonic())
            try:
                if skip_enrich:
                    for seg in j.segments:
                        seg.transcript_enriched = seg.transcript_raw
                elif j.status.get("status") == "transcribed":
                    t0 = time.monotonic()
                    print(f"[{vid}] Enriching...")
                    raw_texts = [s.transcript_raw for s in j.segments]
                    enriched = await asyncio.to_thread(
                        enricher.enrich, j.job.context or j.job.title, raw_texts,
                    )
                    for seg, e in zip(j.segments, enriched):
                        seg.transcript_enriched = e
                    j.status["enriched"] = [s.transcript_enriched for s in j.segments]
                    j.status["status"] = "enriched"
                    _save_status(output_dir, vid, j.status)
                    print(f"[{vid}] Enriched in {time.monotonic() - t0:.0f}s (waited {waited:.0f}s)")
                j.queued_at["embed"] = time.monotonic()
                q_embed.put_nowait(j)
            except Exception as exc:
                await _log_failure(j.job, f"{type(exc).__name__}: {exc}")
                print(f"[{vid}] FAILED: {exc}")
            q_enrich.task_done()

    # --- Stage: Embed (CPU, runs inline) ---
    async def embed_worker():
        while True:
            j = await q_embed.get()
            vid = j.job.video_id
            waited = time.monotonic() - j.queued_at.get("embed", time.monotonic())
            try:
                t0 = time.monotonic()
                print(f"[{vid}] Embedding...")
                texts = [s.transcript_enriched for s in j.segments]
                embeddings = await asyncio.to_thread(embedder.embed_batch, texts)
                for seg, emb in zip(j.segments, embeddings):
                    seg.text_embedding = emb
                j.status["embeddings"] = embeddings
                j.status["status"] = "embedded"
                _save_status(output_dir, vid, j.status)
                print(f"[{vid}] Embedded in {time.monotonic() - t0:.0f}s (waited {waited:.0f}s)")
                j.queued_at["frames"] = time.monotonic()
                q_frames.put_nowait(j)
            except Exception as exc:
                await _log_failure(j.job, f"{type(exc).__name__}: {exc}")
                print(f"[{vid}] FAILED: {exc}")
            q_embed.task_done()

    # --- Stage: Frames + Package ---
    async def frames_worker():
        nonlocal done_count
        while True:
            j = await q_frames.get()
            vid = j.job.video_id
            waited = time.monotonic() - j.queued_at.get("frames", time.monotonic())
            video_path = output_dir / f"{vid}.video"
            try:
                rtt_path = output_dir / f"{vid}.rtt"
                fd = _frames_dir(output_dir, vid)
                fd.mkdir(exist_ok=True)
                timestamps = [s.start_seconds for s in j.segments]
                t0 = time.monotonic()
                if _is_youtube(j):
                    print(f"[{vid}] Downloading video for frames...")
                    dl_path = await asyncio.to_thread(
                        youtube.RealDownloader.download_video, vid, output_dir,
                    )
                    dl_path.rename(video_path)
                    frame_paths = await asyncio.to_thread(
                        frames.extract, video_path, timestamps, fd,
                    )
                    video_path.unlink(missing_ok=True)
                else:
                    print(f"[{vid}] Extracting remote frames...")
                    frame_paths = await frames.extract_remote(
                        j.job.source_url, timestamps, fd,
                    )
                print(f"[{vid}] Frames done in {time.monotonic() - t0:.0f}s (waited {waited:.0f}s)")
                for seg, fp in zip(j.segments, frame_paths):
                    seg.frame_path = f"frames/{fp.name}" if fp else ""

                for seg in j.segments:
                    seg.collection = j.job.collection
                duration = max(s.end_seconds for s in j.segments) if j.segments else 0.0
                video = t.Video(
                    video_id=vid, title=j.job.title,
                    source_url=j.job.page_url or j.job.source_url,
                    context=j.job.context or j.job.title,
                    duration_seconds=duration, status="ready",
                    collection=j.job.collection,
                )
                package.create(video, j.segments, fd, rtt_path)
                _cleanup(output_dir, vid)
                video_path.unlink(missing_ok=True)
                done_count += 1
                print(f"[{vid}] Done ({done_count}/{total}) -> {rtt_path}")
                results.append(rtt_path)
            except Exception as exc:
                await _log_failure(j.job, f"{type(exc).__name__}: {exc}")
                video_path.unlink(missing_ok=True)
                print(f"[{vid}] FAILED: {exc}")
            q_frames.task_done()

    batch_start = time.monotonic()

    async def status_printer():
        while True:
            await asyncio.sleep(10)
            elapsed = time.monotonic() - batch_start
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"[status] queues: transcribe={q_transcribe.qsize()} enrich={q_enrich.qsize()} embed={q_embed.qsize()} frames={q_frames.qsize()} | done={done_count}/{total} | {mins}m{secs}s elapsed")

    workers = []
    workers.append(asyncio.create_task(status_printer()))
    for _ in range(concurrency_transcribe):
        workers.append(asyncio.create_task(transcribe_worker()))
    for _ in range(concurrency_enrich):
        workers.append(asyncio.create_task(enrich_worker()))
    for _ in range(concurrency_embed):
        workers.append(asyncio.create_task(embed_worker()))
    for _ in range(concurrency_frames):
        workers.append(asyncio.create_task(frames_worker()))

    skipped = 0
    deferred_new = []
    for job in jobs:
        rtt_path = output_dir / f"{job.video_id}.rtt"
        if rtt_path.exists():
            print(f"[{job.video_id}] Already packaged, skipping")
            results.append(rtt_path)
            skipped += 1
            continue
        status = _load_status(output_dir, job.video_id)
        if status.get("status") == "new":
            deferred_new.append(job)
        else:
            j = _Job(job=job)
            st = status.get("status")
            j.status = status
            j.segments = [
                t.Segment(segment_id=s["segment_id"], video_id=job.video_id,
                          start_seconds=s["start"], end_seconds=s["end"],
                          transcript_raw=s["text"])
                for s in status["segments"]
            ]
            if st == "transcribed":
                j.queued_at["enrich"] = time.monotonic()
                q_enrich.put_nowait(j)
            elif st in ("enriched", "embedded"):
                for seg, e in zip(j.segments, status.get("enriched", [])):
                    seg.transcript_enriched = e
                if st == "embedded":
                    for seg, emb in zip(j.segments, status.get("embeddings", [])):
                        seg.text_embedding = emb
                    j.queued_at["frames"] = time.monotonic()
                    q_frames.put_nowait(j)
                else:
                    j.queued_at["embed"] = time.monotonic()
                    q_embed.put_nowait(j)
            else:
                deferred_new.append(job)
            print(f"[{job.video_id}] Resuming from {st}")
    for job in deferred_new:
        j = _Job(job=job)
        j.queued_at["transcribe"] = time.monotonic()
        q_transcribe.put_nowait(j)

    await q_transcribe.join()
    await q_enrich.join()
    await q_embed.join()
    await q_frames.join()

    for w in workers:
        w.cancel()

    n_failed = total - len(results)
    print(f"\nBatch complete: {len(results)}/{total} succeeded ({skipped} skipped)")
    if n_failed > 0:
        print(f"Failures logged to {fail_path}")
    return results
