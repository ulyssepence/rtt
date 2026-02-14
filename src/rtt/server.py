import tempfile
import zipfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rtt import embed, package, vector


class SegmentResult(BaseModel):
    video_id: str
    segment_id: str
    start_seconds: float
    end_seconds: float
    source_url: str
    title: str
    transcript_raw: str
    transcript_enriched: str
    frame_url: str | None = None
    page_url: str | None = None
    collection: str = ""
    score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    results: list[SegmentResult]


class SegmentsResponse(BaseModel):
    segments: list[SegmentResult]
    total: int
    offset: int
    limit: int


class CollectionInfo(BaseModel):
    id: str
    video_count: int
    segment_count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


def _collect_rtt_files(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for p in paths:
        if p.is_dir():
            result.extend(sorted(p.glob("*.rtt")))
        elif p.suffix == ".rtt" and p.exists():
            result.append(p)
    return result


def create_app(rtt_paths: Path | list[Path], embedder: embed.Embedder | None = None) -> FastAPI:
    app = FastAPI(title="RTT Semantic Video Search")
    db = vector.Database.memory()
    _embedder = embedder or embed.OllamaEmbedder()
    videos: dict[str, dict] = {}
    frames_dir = Path(tempfile.mkdtemp(prefix="rtt_frames_"))

    if isinstance(rtt_paths, Path):
        rtt_paths = [rtt_paths]
    for rtt_path in _collect_rtt_files(rtt_paths):
        vid, segments, arrow_table = package.load(rtt_path)

        embeddings = arrow_table.column("text_embedding").to_pylist()
        bad = [e for e in embeddings if len(e) != 768]
        if bad:
            print(f"Skipping {rtt_path.name}: {len(bad)}/{len(embeddings)} embeddings have wrong dimensions ({set(len(e) for e in bad)})")
            continue

        videos[vid.video_id] = {
            "title": vid.title,
            "remote_url": vid.source_url or None,
            "page_url": vid.page_url or None,
            "collection": vid.collection,
            "local_dir": rtt_path.parent,
        }

        seg_objects = []
        for seg, emb in zip(segments, embeddings):
            seg.text_embedding = emb
            seg_objects.append(seg)
        db.add(seg_objects)

        vid_frames = frames_dir / vid.video_id
        vid_frames.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(rtt_path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("frames/") and name.endswith(".jpg"):
                    data = zf.read(name)
                    (vid_frames / Path(name).name).write_bytes(data)

    app.mount("/frames", StaticFiles(directory=str(frames_dir)), name="frames")

    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="static")

    frontend_index = Path(__file__).parent.parent.parent / "frontend" / "index.html"

    _http_client = httpx.Client(follow_redirects=True, timeout=30)

    def _to_result(r: dict, score: float = 0.0) -> SegmentResult:
        vid_id = r["video_id"]
        vid_info = videos.get(vid_id, {})
        frame_path = r.get("frame_path", "")
        frame_url = f"/frames/{vid_id}/{Path(frame_path).name}" if frame_path else None
        return SegmentResult(
            video_id=vid_id,
            segment_id=r["segment_id"],
            start_seconds=r["start_seconds"],
            end_seconds=r["end_seconds"],
            source_url=f"/video/{vid_id}",
            title=vid_info.get("title", ""),
            transcript_raw=r.get("transcript_raw", ""),
            transcript_enriched=r.get("transcript_enriched", ""),
            frame_url=frame_url,
            page_url=vid_info.get("page_url"),
            collection=vid_info.get("collection", ""),
            score=score,
        )

    @app.get("/video/{video_id}")
    def video(video_id: str, request: Request):
        vid_info = videos.get(video_id)
        if not vid_info:
            raise HTTPException(status_code=404, detail="Video not found")
        local_dir = vid_info["local_dir"]
        for ext in (".mp4", ".webm", ".mkv"):
            candidate = local_dir / f"{video_id}{ext}"
            if candidate.exists():
                return FileResponse(str(candidate), media_type=f"video/{ext[1:]}")
        remote_url = vid_info.get("remote_url")
        if not remote_url:
            raise HTTPException(status_code=404, detail="Video file not found")
        headers = {}
        if "range" in request.headers:
            headers["range"] = request.headers["range"]
        upstream = _http_client.send(_http_client.build_request("GET", remote_url, headers=headers), stream=True)
        resp_headers = {}
        for key in ("content-length", "content-range", "accept-ranges"):
            if key in upstream.headers:
                resp_headers[key] = upstream.headers[key]
        return StreamingResponse(
            upstream.iter_bytes(chunk_size=64 * 1024),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "video/mp4"),
            headers=resp_headers,
        )

    @app.get("/")
    def index():
        if frontend_index.exists():
            return FileResponse(str(frontend_index))
        return JSONResponse({"error": "Frontend not built"}, status_code=404)

    @app.get("/search", response_model=SearchResponse)
    def search(
        q: str = Query(default=""),
        segment_id: str = Query(default=""),
        collections: str = Query(default=""),
        n: int = Query(default=50, ge=1, le=200),
    ):
        col_filter = [c for c in collections.split(",") if c] if collections else None

        if segment_id:
            seg = db.get_segment(segment_id)
            if not seg:
                raise HTTPException(status_code=404, detail="Segment not found")
            query_vec = seg["text_embedding"]
            raw = db.closest(query_vec, n=n, collections=col_filter)
            results = [_to_result(r, r.get("_distance", 0.0)) for r in raw]
            return SearchResponse(query=f"similar:{segment_id}", results=results)

        if not q.strip():
            raise HTTPException(status_code=400, detail="Empty query")

        query_vec = _embedder.embed(q)
        raw = db.closest(query_vec, n=n, collections=col_filter)
        results = [_to_result(r, r.get("_distance", 0.0)) for r in raw]
        return SearchResponse(query=q, results=results)

    @app.get("/segments", response_model=SegmentsResponse)
    def segments(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
        collections: str = Query(default=""),
    ):
        col_filter = [c for c in collections.split(",") if c] if collections else None
        rows = db.list_segments(offset=offset, limit=limit, collections=col_filter)
        total = db.count(collections=col_filter)
        results = [_to_result(r) for r in rows]
        return SegmentsResponse(segments=results, total=total, offset=offset, limit=limit)

    @app.get("/collections", response_model=CollectionsResponse)
    def collections_list():
        col_data: dict[str, dict] = {}
        for vid_id, info in videos.items():
            col = info.get("collection", "") or ""
            if col not in col_data:
                col_data[col] = {"videos": set(), "segment_count": 0}
            col_data[col]["videos"].add(vid_id)
        for col in col_data:
            col_data[col]["segment_count"] = db.count(collections=[col])
        result = [
            CollectionInfo(id=col_id, video_count=len(info["videos"]), segment_count=info["segment_count"])
            for col_id, info in sorted(col_data.items())
        ]
        return CollectionsResponse(collections=result)

    return app
