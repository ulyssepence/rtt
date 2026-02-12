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


@pytest.fixture
def rtt_dir():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        frames_dir = tmp / "build_frames"
        frames_dir.mkdir()
        (frames_dir / "000000.jpg").write_bytes(b"\xff\xd8fake")
        (frames_dir / "000005.jpg").write_bytes(b"\xff\xd8fake")

        (tmp / "test.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom")

        video = t.Video(
            video_id="test", title="Test", source_url="",
            context="Test", duration_seconds=10.0,
        )
        segments = [
            t.Segment(
                segment_id="test_00000", video_id="test",
                start_seconds=0.0, end_seconds=4.0,
                transcript_raw="nuclear bomb safety",
                transcript_enriched="nuclear bomb safety enriched",
                text_embedding=[1.0] + [0.0] * 767,
                frame_path="frames/000000.jpg",
            ),
            t.Segment(
                segment_id="test_00001", video_id="test",
                start_seconds=5.0, end_seconds=9.0,
                transcript_raw="chocolate cake recipe",
                transcript_enriched="chocolate cake recipe enriched",
                text_embedding=[0.0] * 767 + [1.0],
                frame_path="frames/000005.jpg",
            ),
        ]

        package.create(video, segments, frames_dir, tmp / "test.rtt")
        yield tmp


@pytest.fixture
def client(rtt_dir):
    from rtt import server
    app = server.create_app(rtt_dir, embedder=FakeEmbedder())
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


def test_remote_source_url_passed_through():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames_dir = tmp / "build_frames"
        frames_dir.mkdir()
        (frames_dir / "000000.jpg").write_bytes(b"\xff\xd8fake")
        video = t.Video(
            video_id="remote", title="Remote", source_url="https://archive.org/download/test/test.mp4",
            context="Remote", duration_seconds=5.0,
        )
        segments = [t.Segment(
            segment_id="remote_00000", video_id="remote",
            start_seconds=0.0, end_seconds=5.0,
            transcript_raw="test", transcript_enriched="test",
            text_embedding=[1.0] + [0.0] * 767, frame_path="frames/000000.jpg",
        )]
        package.create(video, segments, frames_dir, tmp / "remote.rtt")
        from rtt import server
        app = server.create_app(tmp, embedder=FakeEmbedder())
        client = TestClient(app)
        resp = client.get("/search?q=nuclear+bomb")
        r = resp.json()["results"][0]
        assert r["source_url"] == "https://archive.org/download/test/test.mp4"
