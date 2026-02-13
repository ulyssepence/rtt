import json
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rtt import types as t, package, embed


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for text in texts:
            if "nuclear" in text.lower() or "bomb" in text.lower():
                vecs.append([1.0] + [0.0] * 767)
            else:
                vecs.append([0.0] * 767 + [1.0])
        return vecs


def _make_rtt(tmp: Path, video_id="test", title="Test", source_url="",
              page_url="", collection="", segments_data=None):
    frames_dir = tmp / f"{video_id}_frames"
    frames_dir.mkdir(exist_ok=True)

    if segments_data is None:
        (frames_dir / "000000.jpg").write_bytes(b"\xff\xd8fake")
        (frames_dir / "000005.jpg").write_bytes(b"\xff\xd8fake")
        segments_data = [
            dict(segment_id=f"{video_id}_00000", start=0.0, end=4.0,
                 raw="nuclear bomb safety", enriched="nuclear bomb safety enriched",
                 emb=[1.0] + [0.0] * 767, frame="frames/000000.jpg"),
            dict(segment_id=f"{video_id}_00001", start=5.0, end=9.0,
                 raw="chocolate cake recipe", enriched="chocolate cake recipe enriched",
                 emb=[0.0] * 767 + [1.0], frame="frames/000005.jpg"),
        ]

    video = t.Video(
        video_id=video_id, title=title, source_url=source_url,
        context=title, duration_seconds=10.0, collection=collection,
        page_url=page_url,
    )
    segments = [
        t.Segment(
            segment_id=s["segment_id"], video_id=video_id,
            start_seconds=s["start"], end_seconds=s["end"],
            transcript_raw=s["raw"], transcript_enriched=s["enriched"],
            text_embedding=s["emb"], frame_path=s["frame"],
            collection=collection,
        )
        for s in segments_data
    ]
    package.create(video, segments, frames_dir, tmp / f"{video_id}.rtt")


@pytest.fixture
def rtt_dir():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "test.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom")
        _make_rtt(tmp, video_id="test", title="Test", collection="prelinger")
        yield tmp


@pytest.fixture
def multi_collection_dir():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_rtt(tmp, video_id="vid1", title="Video 1", collection="prelinger")
        _make_rtt(tmp, video_id="vid2", title="Video 2", collection="youtube")
        yield tmp


@pytest.fixture
def client(rtt_dir):
    from rtt import server
    app = server.create_app(rtt_dir, embedder=FakeEmbedder())
    return TestClient(app)


@pytest.fixture
def multi_client(multi_collection_dir):
    from rtt import server
    app = server.create_app(multi_collection_dir, embedder=FakeEmbedder())
    return TestClient(app)


def test_search_returns_results(client):
    resp = client.get("/search?q=nuclear+bomb")
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "nuclear bomb"
    assert len(data["results"]) > 0

    r = data["results"][0]
    assert "video_id" in r
    assert "segment_id" in r
    assert "start_seconds" in r
    assert "end_seconds" in r
    assert "source_url" in r
    assert "title" in r
    assert r["start_seconds"] >= 0


def test_search_ranking(client):
    resp = client.get("/search?q=nuclear+bomb")
    results = resp.json()["results"]
    assert results[0]["segment_id"] == "test_00000"


def test_empty_query_400(client):
    resp = client.get("/search?q=")
    assert resp.status_code == 400


def test_frame_url_resolves(client):
    resp = client.get("/search?q=nuclear+bomb")
    results = resp.json()["results"]
    frame_url = results[0].get("frame_url")
    if frame_url:
        img_resp = client.get(frame_url)
        assert img_resp.status_code == 200


def test_source_url_resolves_for_local_video(client):
    resp = client.get("/search?q=nuclear+bomb")
    r = resp.json()["results"][0]
    assert r["source_url"].startswith("/video/")
    video_resp = client.get(r["source_url"])
    assert video_resp.status_code == 200


def test_video_endpoint_404_for_unknown(client):
    resp = client.get("/video/nonexistent")
    assert resp.status_code == 404


def test_source_url_always_uses_video_route():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_rtt(tmp, video_id="remote", title="Remote",
                  source_url="https://archive.org/download/test/test.mp4",
                  segments_data=[
                      dict(segment_id="remote_00000", start=0.0, end=5.0,
                           raw="test", enriched="test",
                           emb=[1.0] + [0.0] * 767, frame="frames/000000.jpg"),
                  ])
        frames_dir = tmp / "remote_frames"
        frames_dir.mkdir(exist_ok=True)
        (frames_dir / "000000.jpg").write_bytes(b"\xff\xd8fake")
        from rtt import server
        app = server.create_app(tmp, embedder=FakeEmbedder())
        client = TestClient(app)
        resp = client.get("/search?q=nuclear+bomb")
        r = resp.json()["results"][0]
        assert r["source_url"] == "/video/remote"


def test_segments_endpoint(client):
    resp = client.get("/segments?offset=0&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "segments" in data
    assert data["total"] == 2
    assert len(data["segments"]) == 2
    assert data["offset"] == 0
    assert data["limit"] == 10


def test_segments_pagination(client):
    resp = client.get("/segments?offset=0&limit=1")
    data = resp.json()
    assert len(data["segments"]) == 1
    resp2 = client.get("/segments?offset=1&limit=1")
    data2 = resp2.json()
    assert len(data2["segments"]) == 1
    assert data["segments"][0]["segment_id"] != data2["segments"][0]["segment_id"]


def test_collections_endpoint(client):
    resp = client.get("/collections")
    assert resp.status_code == 200
    data = resp.json()
    assert "collections" in data
    assert len(data["collections"]) >= 1


def test_collections_multi(multi_client):
    resp = multi_client.get("/collections")
    data = resp.json()
    ids = {c["id"] for c in data["collections"]}
    assert "prelinger" in ids
    assert "youtube" in ids


def test_search_by_segment_id(client):
    resp = client.get("/search?segment_id=test_00000")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) > 0
    assert data["query"] == "similar:test_00000"


def test_search_by_segment_id_not_found(client):
    resp = client.get("/search?segment_id=nonexistent")
    assert resp.status_code == 404


def test_search_collection_filter(multi_client):
    resp = multi_client.get("/search?q=nuclear+bomb&collections=prelinger")
    data = resp.json()
    for r in data["results"]:
        assert r["collection"] == "prelinger"


def test_segments_collection_filter(multi_client):
    resp = multi_client.get("/segments?collections=youtube")
    data = resp.json()
    for s in data["segments"]:
        assert s["collection"] == "youtube"


def test_search_result_has_collection(client):
    resp = client.get("/search?q=nuclear+bomb")
    r = resp.json()["results"][0]
    assert "collection" in r
    assert r["collection"] == "prelinger"
