import time
import zipfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pyarrow as pa
import pyarrow.compute as pc

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
    context: str = ""
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
            result.extend(sorted(p.glob("**/*.rtt")))
        elif p.suffix == ".rtt" and p.exists():
            result.append(p)
    return result


def create_app(rtt_paths: Path | list[Path], embedder: embed.Embedder | None = None) -> FastAPI:
    app = FastAPI(title="RTT Semantic Video Search")
    db = vector.Database.memory()
    _embedder = embedder or embed.OllamaEmbedder()
    videos: dict[str, dict] = {}
    rtt_paths_by_video: dict[str, Path] = {}

    def _mem_mb() -> int:
        import resource, sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss // 1024 if sys.platform == "linux" else rss // (1024 * 1024)

    t0 = time.monotonic()
    total_segments = 0
    if isinstance(rtt_paths, Path):
        rtt_paths = [rtt_paths]
    rtt_files = _collect_rtt_files(rtt_paths)
    print(f"Found {len(rtt_files)} .rtt files, RSS={_mem_mb()}MB")
    for i, rtt_path in enumerate(rtt_files):
        if i % 100 == 0:
            print(f"  loading {i}/{len(rtt_files)} RSS={_mem_mb()}MB")
        vid, arrow_table = package.load_metadata(rtt_path)

        emb_type = arrow_table.schema.field("text_embedding").type
        if hasattr(emb_type, "list_size") and emb_type.list_size != 768:
            print(f"Skipping {rtt_path.name}: embeddings have dimension {emb_type.list_size}, expected 768")
            continue
        if not hasattr(emb_type, "list_size"):
            lengths = pc.list_value_length(arrow_table.column("text_embedding"))
            bad_count = pc.sum(pc.not_equal(lengths, 768)).as_py()
            if bad_count:
                print(f"Skipping {rtt_path.name}: {bad_count}/{len(arrow_table)} embeddings have wrong dimensions")
                continue

        videos[vid.video_id] = {
            "title": vid.title,
            "remote_url": vid.source_url or None,
            "page_url": vid.page_url or None,
            "collection": vid.collection,
            "context": vid.context or "",
            "local_dir": rtt_path.parent,
        }
        rtt_paths_by_video[vid.video_id] = rtt_path
        db.add_table(arrow_table)
        total_segments += len(arrow_table)

    t_load = time.monotonic()
    print(f"Loaded {len(videos)} files, {total_segments} segments in {(t_load - t0) * 1000:.0f}ms, RSS={_mem_mb()}MB")

    db._ensure_merged()
    t_merge = time.monotonic()
    print(f"Merged search index in {(t_merge - t_load) * 1000:.0f}ms, RSS={_mem_mb()}MB")

    db.compact()
    print(f"Compacted, RSS={_mem_mb()}MB")

    frontend_index = Path(__file__).parent.parent.parent / "frontend" / "index.html"

    _http_client = httpx.Client(follow_redirects=True, timeout=30)
    _resolved_urls: dict[str, str] = {}

    def _to_result(r: dict, score: float = 0.0) -> SegmentResult:
        vid_id = r["video_id"]
        vid_info = videos.get(vid_id, {})
        frame_path = r.get("frame_path", "")
        frame_url = f"/static/frames/{vid_id}/{Path(frame_path).name}" if frame_path else None
        return SegmentResult(
            video_id=vid_id,
            segment_id=r["segment_id"],
            start_seconds=r["start_seconds"],
            end_seconds=r["end_seconds"],
            source_url=vid_info.get("remote_url") or f"/video/{vid_id}",
            title=vid_info.get("title", ""),
            transcript_raw=r.get("transcript_raw", ""),
            transcript_enriched=r.get("transcript_enriched", ""),
            frame_url=frame_url,
            page_url=vid_info.get("page_url"),
            collection=vid_info.get("collection", ""),
            context=vid_info.get("context", ""),
            score=score,
        )

    @app.get("/static/frames/{video_id}/{filename}")
    def frame(video_id: str, filename: str):
        rtt_path = rtt_paths_by_video.get(video_id)
        if not rtt_path:
            raise HTTPException(status_code=404, detail="Video not found")
        try:
            with zipfile.ZipFile(rtt_path, "r") as zf:
                data = zf.read(f"frames/{filename}")
        except KeyError:
            raise HTTPException(status_code=404, detail="Frame not found")
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/video/{video_id}/resolve")
    def resolve_video(video_id: str):
        vid_info = videos.get(video_id)
        if not vid_info:
            raise HTTPException(status_code=404, detail="Video not found")
        if video_id in _resolved_urls:
            return {"url": _resolved_urls[video_id]}
        remote_url = vid_info.get("remote_url")
        if not remote_url:
            return {"url": f"/video/{video_id}"}
        try:
            resp = httpx.head(remote_url, follow_redirects=True, timeout=10)
            final = str(resp.url)
            _resolved_urls[video_id] = final
            return {"url": final}
        except Exception:
            return {"url": f"/video/{video_id}"}

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

    @app.get("/static/video/{video_id}/segments")
    def video_segments(video_id: str):
        if video_id not in videos:
            raise HTTPException(status_code=404, detail="Video not found")
        rows = db.video_segments(video_id)
        results = [_to_result(r) for r in rows]
        return JSONResponse(
            content=[r.model_dump() for r in results],
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/segments", response_model=SegmentsResponse)
    @app.get("/static/segments", response_model=SegmentsResponse)
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

    frontend_static = Path(__file__).parent.parent.parent / "frontend" / "static"
    if frontend_static.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_static)), name="static")

    return app
