import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from rtt import batch, types as t


def _mock_assemblyai_transcriber():
    transcriber = MagicMock()
    transcriber.transcribe_url.return_value = [
        t.Segment(
            segment_id="test_00000", video_id="test",
            start_seconds=0.5, end_seconds=2.0,
            transcript_raw="Duck and cover.",
        ),
        t.Segment(
            segment_id="test_00001", video_id="test",
            start_seconds=3.0, end_seconds=6.5,
            transcript_raw="When you see the flash, duck and cover.",
        ),
        t.Segment(
            segment_id="test_00002", video_id="test",
            start_seconds=10.0, end_seconds=12.0,
            transcript_raw="This is the end of the film.",
        ),
    ]
    return transcriber


IA_SAMPLE_URL = "https://archive.org/download/DuckandC1951/DuckandC1951_512kb.mp4"


def test_batch_single_video_mocked_transcription():
    job = t.VideoJob(
        video_id="duck_and_cover",
        title="Duck and Cover",
        source_url=IA_SAMPLE_URL,
        context="Cold War civil defense film",
    )

    mock_transcriber = _mock_assemblyai_transcriber()

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)

        with patch.object(batch, "transcribe") as mock_mod:
            mock_mod.AssemblyAITranscriber.return_value = mock_transcriber
            paths = asyncio.run(batch.process_batch(
                [job], output_dir, skip_enrich=True,
            ))

        assert len(paths) == 1
        rtt_path = paths[0]
        assert rtt_path.exists()
        assert rtt_path.suffix == ".rtt"

        from rtt import package
        video, segments, table = package.load(rtt_path)
        assert video.video_id == "duck_and_cover"
        assert video.title == "Duck and Cover"
        assert len(segments) == 3
        assert table.num_rows == 3
        assert len(table.column("text_embedding")[0].as_py()) == 768
