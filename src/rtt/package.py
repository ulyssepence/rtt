import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rtt import types as t, vector


def create(video: t.Video, segments: list[t.Segment], frames_dir: Path | None, output_path: Path) -> Path:
    table = pa.table({
        "segment_id": [s.segment_id for s in segments],
        "video_id": [s.video_id for s in segments],
        "start_seconds": [s.start_seconds for s in segments],
        "end_seconds": [s.end_seconds for s in segments],
        "transcript_raw": [s.transcript_raw for s in segments],
        "transcript_enriched": [s.transcript_enriched for s in segments],
        "text_embedding": [s.text_embedding for s in segments],
        "frame_path": [s.frame_path for s in segments],
        "has_speech": [s.has_speech for s in segments],
        "source": [s.source for s in segments],
    })

    manifest = {
        "video_id": video.video_id,
        "status": "ready",
        "title": video.title,
        **({"source_url": video.source_url} if video.source_url else {}),
        **({"page_url": video.page_url} if video.page_url else {}),
        "context": video.context,
        "duration_seconds": video.duration_seconds,
        "segments": [
            {
                "segment_id": s.segment_id,
                "start_seconds": s.start_seconds,
                "end_seconds": s.end_seconds,
                "source": s.source,
                "transcript_raw": s.transcript_raw,
                "transcript_enriched": s.transcript_enriched,
                "frame_path": s.frame_path,
                "has_speech": s.has_speech,
            }
            for s in segments
        ],
    }

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        pq_buf = pa.BufferOutputStream()
        pq.write_table(table, pq_buf)
        zf.writestr("segments.parquet", pq_buf.getvalue().to_pybytes())

        if frames_dir and frames_dir.exists():
            for frame in sorted(frames_dir.glob("*.jpg")):
                zf.write(frame, f"frames/{frame.name}")

    return output_path


def load(rtt_path: Path) -> tuple[t.Video, list[t.Segment], pa.Table]:
    with zipfile.ZipFile(rtt_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        pq_bytes = zf.read("segments.parquet")

    video = t.Video(
        video_id=manifest["video_id"],
        title=manifest["title"],
        source_url=manifest.get("source_url", ""),
        page_url=manifest.get("page_url", ""),
        context=manifest["context"],
        duration_seconds=manifest["duration_seconds"],
        status=manifest["status"],
    )

    segments = [
        t.Segment(
            segment_id=s["segment_id"],
            video_id=manifest["video_id"],
            start_seconds=s["start_seconds"],
            end_seconds=s["end_seconds"],
            transcript_raw=s["transcript_raw"],
            transcript_enriched=s["transcript_enriched"],
            frame_path=s.get("frame_path", ""),
            has_speech=s.get("has_speech", True),
            source=s.get("source", "transcript"),
        )
        for s in manifest["segments"]
    ]

    buf = pa.py_buffer(pq_bytes)
    table = pq.read_table(pa.BufferReader(buf))

    return video, segments, table
