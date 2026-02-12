import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
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
        source_url = vid.source_url
        if not source_url:
            for ext in (".mp4", ".webm", ".mkv"):
                candidate = rtt_path.with_suffix(ext)
                if candidate.exists():
                    source_url = f"/video/{vid.video_id}"
                    break
        videos[vid.video_id] = {"title": vid.title, "source_url": source_url, "local_path": rtt_path.parent}

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

    @app.get("/video/{video_id}")
    def video(video_id: str):
        vid_info = videos.get(video_id)
        if not vid_info:
            raise HTTPException(status_code=404, detail="Video not found")
        local_path = vid_info.get("local_path")
        if not local_path:
            raise HTTPException(status_code=404, detail="No local video")
        for ext in (".mp4", ".webm", ".mkv"):
            candidate = local_path / f"{video_id}{ext}"
            if candidate.exists():
                return FileResponse(str(candidate), media_type=f"video/{ext[1:]}")
        raise HTTPException(status_code=404, detail="Video file not found")

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
                source_url=vid_info.get("source_url", ""),
                title=vid_info.get("title", ""),
                transcript_raw=r.get("transcript_raw", ""),
                transcript_enriched=r.get("transcript_enriched", ""),
                frame_url=frame_url,
                score=r.get("_distance", 0.0),
            ))

        return SearchResponse(query=q, results=results)

    return app
