import asyncio
import json
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

from rtt import types as t, transcribe, enrich, embed, frames, package

TRANSCRIBE_CONCURRENCY = 50
ENRICH_CONCURRENCY = 5


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
        rtt_path = output_dir / f"{job.video_id}.rtt"
        if rtt_path.exists():
            print(f"[{job.video_id}] Already exists, skipping")
            return rtt_path

        try:
            print(f"[{job.video_id}] Transcribing...")
            async with sem_transcribe:
                segments = await asyncio.to_thread(
                    transcriber.transcribe_url, job.source_url, job.video_id,
                )
            print(f"[{job.video_id}] Transcribed ({len(segments)} segments)")

            if not segments:
                await _log_failure(job, "No segments returned (silent video?)")
                print(f"[{job.video_id}] FAILED: no segments")
                return None

            if skip_enrich:
                for seg in segments:
                    seg.transcript_enriched = seg.transcript_raw
            else:
                print(f"[{job.video_id}] Enriching...")
                async with sem_enrich:
                    raw_texts = [s.transcript_raw for s in segments]
                    enriched = await asyncio.to_thread(
                        enricher.enrich, job.context or job.title, raw_texts,
                    )
                    for seg, e in zip(segments, enriched):
                        seg.transcript_enriched = e
                print(f"[{job.video_id}] Enriched")

            print(f"[{job.video_id}] Embedding...")
            texts = [s.transcript_enriched for s in segments]
            embeddings = await asyncio.to_thread(embedder.embed_batch, texts)
            for seg, emb in zip(segments, embeddings):
                seg.text_embedding = emb
            print(f"[{job.video_id}] Embedded")

            print(f"[{job.video_id}] Extracting frames...")
            with tempfile.TemporaryDirectory(prefix=f"rtt_frames_{job.video_id}_") as tmp:
                frames_dir = Path(tmp)
                timestamps = [s.start_seconds for s in segments]
                frame_paths = await frames.extract_remote(
                    job.source_url, timestamps, frames_dir,
                )
                for seg, fp in zip(segments, frame_paths):
                    seg.frame_path = f"frames/{fp.name}" if fp else ""

                duration = max(s.end_seconds for s in segments) if segments else 0.0
                video = t.Video(
                    video_id=job.video_id, title=job.title,
                    source_url=job.source_url, context=job.context or job.title,
                    duration_seconds=duration, status="ready",
                )

                package.create(video, segments, frames_dir, rtt_path)

            print(f"[{job.video_id}] Done -> {rtt_path}")
            return rtt_path

        except Exception as exc:
            await _log_failure(job, f"{type(exc).__name__}: {exc}")
            print(f"[{job.video_id}] FAILED: {exc}")
            return None

    tasks = [_process_one(job) for job in jobs]
    results = await asyncio.gather(*tasks)
    paths = [p for p in results if p is not None]
    n_failed = len(jobs) - len(paths)
    print(f"\nBatch complete: {len(paths)}/{len(jobs)} succeeded")
    if n_failed > 0:
        print(f"Failures logged to {fail_path}")
    return paths
