import shutil
import tempfile
from pathlib import Path

import pytest

from rtt import main as pipeline, package, vector, embed, types as t

_rtt_cache: Path | None = None
_tmp_dir = None


@pytest.fixture(scope="module")
def rtt_file(request):
    global _rtt_cache, _tmp_dir
    if _rtt_cache and _rtt_cache.exists():
        return _rtt_cache

    sample_video = Path(__file__).parent.parent / "data" / "sample" / "KnifeThr1950_512kb.mp4"
    if not sample_video.exists():
        pytest.skip("data/sample/KnifeThr1950_512kb.mp4 not found")

    _tmp_dir = tempfile.mkdtemp()
    tmp = Path(_tmp_dir)
    vid = tmp / sample_video.name
    shutil.copy2(sample_video, vid)
    _rtt_cache = pipeline.process(vid, video_id="e2e_test", title="E2E Test")
    return _rtt_cache


def test_pipeline_produces_valid_rtt(rtt_file):
    video, segments, arrow_table = package.load(rtt_file)
    assert video.status == "ready"
    assert len(segments) > 0
    assert len(arrow_table) > 0
    embs = arrow_table.column("text_embedding").to_pylist()
    assert len(embs[0]) == 768


def test_search_with_decoys(rtt_file):
    embedder = embed.OllamaEmbedder()
    video, segments, arrow_table = package.load(rtt_file)

    db = vector.Database.memory()
    loaded_segs = []
    embeddings = arrow_table.column("text_embedding").to_pylist()
    for seg, emb in zip(segments, embeddings):
        seg.text_embedding = emb
        loaded_segs.append(seg)
    db.add(loaded_segs)

    decoy_texts = [
        "how to bake a perfect sourdough bread",
        "basketball championship final score",
        "tropical weather forecast for hawaii",
    ]
    decoy_embs = embedder.embed_batch(decoy_texts)
    decoys = [
        t.Segment(
            segment_id=f"decoy_{i}", video_id="decoy",
            start_seconds=0.0, end_seconds=5.0,
            transcript_raw=text, transcript_enriched=text,
            text_embedding=emb,
        )
        for i, (text, emb) in enumerate(zip(decoy_texts, decoy_embs))
    ]
    db.add(decoys)

    queries = ["children party manners", "etiquette for kids", "social behavior"]
    for q in queries:
        qvec = embedder.embed(q)
        results = db.closest(qvec, n=5)
        assert len(results) > 0
        top = results[0]
        assert top["video_id"] == "e2e_test", f"Query '{q}' returned decoy first: {top['video_id']}"
