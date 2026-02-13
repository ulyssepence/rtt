import tempfile
import zipfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rtt import embed, package, vector


class SearchResult(BaseModel):
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
    score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


def create_app(rtt_dir: Path, embedder: embed.Embedder | None = None) -> FastAPI:
    app = FastAPI(title="RTT Semantic Video Search")
    db = vector.Database.memory()
    _embedder = embedder or embed.OllamaEmbedder()
    videos: dict[str, dict] = {}
    frames_dir = Path(tempfile.mkdtemp(prefix="rtt_frames_"))

    for rtt_path in sorted(rtt_dir.glob("*.rtt")):
        vid, segments, arrow_table = package.load(rtt_path)
        videos[vid.video_id] = {
            "title": vid.title,
            "remote_url": vid.source_url or None,
            "page_url": vid.page_url or None,
            "local_dir": rtt_path.parent,
        }

        seg_objects = []
        embeddings = arrow_table.column("text_embedding").to_pylist()
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
    def search(q: str = Query(default="")):
        if not q.strip():
            raise HTTPException(status_code=400, detail="Empty query")

        query_vec = _embedder.embed(q)
        raw = db.closest(query_vec, n=20)

        results = []
        for r in raw:
            vid_id = r["video_id"]
            vid_info = videos.get(vid_id, {})
            frame_path = r.get("frame_path", "")
            frame_url = f"/frames/{vid_id}/{Path(frame_path).name}" if frame_path else None

            results.append(SearchResult(
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
                score=r.get("_distance", 0.0),
            ))

        return SearchResponse(query=q, results=results)

    return app
