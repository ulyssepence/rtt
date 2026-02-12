from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest
from rtt import transcribe


@pytest.fixture
def transcriber():
    return transcribe.WhisperTranscriber()


def test_transcriber_segment_shape(transcriber, sample_video):
    segments = transcriber.transcribe(sample_video, "test_video")

    assert len(segments) > 0

    for seg in segments:
        assert seg.start_seconds < seg.end_seconds
        assert seg.transcript_raw.strip() != ""
        assert seg.video_id == "test_video"
        assert seg.segment_id.startswith("test_video_")

    for a, b in zip(segments, segments[1:]):
        assert b.start_seconds >= a.start_seconds

    assert segments[0].start_seconds < 5.0


def _mock_assemblyai():
    aai = MagicMock()
    aai.TranscriptStatus.error = "error"

    transcript = MagicMock()
    transcript.status = "completed"
    transcript.error = None
    transcript.utterances = [
        SimpleNamespace(text="Duck and cover.", start=500, end=2000),
        SimpleNamespace(text="When you see the flash, duck and cover.", start=3000, end=6500),
        SimpleNamespace(text="This is the end of the film.", start=10000, end=12000),
    ]
    transcript.words = None
    aai.Transcriber.return_value.transcribe.return_value = transcript
    return aai


@patch.dict("os.environ", {"ASSEMBLYAI_API_KEY": "test-key"})
@patch("assemblyai.settings", new_callable=lambda: type("S", (), {"api_key": None}))
def test_assemblyai_transcriber_segment_shape(_mock_settings):
    aai = _mock_assemblyai()
    with patch.dict("sys.modules", {"assemblyai": aai}):
        t = transcribe.AssemblyAITranscriber(api_key="test-key")
        t._aai = aai

    segments = t.transcribe_url("https://example.com/video.mp4", "test_video")

    assert len(segments) == 3

    for seg in segments:
        assert seg.start_seconds < seg.end_seconds
        assert seg.transcript_raw.strip() != ""
        assert seg.video_id == "test_video"
        assert seg.segment_id.startswith("test_video_")

    for a, b in zip(segments, segments[1:]):
        assert b.start_seconds >= a.start_seconds

    assert abs(segments[0].start_seconds - 0.5) < 0.01
    assert abs(segments[0].end_seconds - 2.0) < 0.01
