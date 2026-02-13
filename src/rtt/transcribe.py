import os
import sys
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from rtt import runtime, types as t


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, video_path: Path, video_id: str) -> list[t.Segment]: ...


class WhisperTranscriber:
    def __init__(self, model=None):
        self._model = model or runtime.ensure_whisper()

    def transcribe(self, video_path: Path, video_id: str) -> list[t.Segment]:
        raw_segments, info = self._model.transcribe(str(video_path), language="en")
        duration = info.duration
        segments = []
        for i, seg in enumerate(raw_segments):
            text = seg.text.strip()
            if not text:
                continue
            segments.append(t.Segment(
                segment_id=f"{video_id}_{i:05d}",
                video_id=video_id,
                start_seconds=seg.start,
                end_seconds=seg.end,
                transcript_raw=text,
            ))
            if duration > 0:
                pct = min(seg.end / duration * 100, 100)
                print(f"\r  Transcribing: {pct:.0f}% ({seg.end:.0f}/{duration:.0f}s)", end="", flush=True)
        if duration > 0:
            print()
        return segments


class AssemblyAITranscriber:
    def __init__(self, api_key: str | None = None):
        import assemblyai as aai
        aai.settings.api_key = api_key or os.environ["ASSEMBLYAI_API_KEY"]
        self._aai = aai

    def transcribe_url(self, url: str, video_id: str) -> list[t.Segment]:
        config = self._aai.TranscriptionConfig(
            speech_model=self._aai.SpeechModel.best,
            filter_profanity=False,
            speaker_labels=False,
            auto_chapters=False,
            entity_detection=False,
            sentiment_analysis=False,
            auto_highlights=False,
            iab_categories=False,
            content_safety=False,
            summarization=False,
        )
        transcript = self._aai.Transcriber().transcribe(url, config=config)
        if transcript.status == self._aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
        segments = []
        for i, utt in enumerate(transcript.utterances or []):
            text = utt.text.strip()
            if not text:
                continue
            segments.append(t.Segment(
                segment_id=f"{video_id}_{i:05d}",
                video_id=video_id,
                start_seconds=utt.start / 1000.0,
                end_seconds=utt.end / 1000.0,
                transcript_raw=text,
            ))
        if not segments and transcript.words:
            segments = self._segments_from_words(transcript.words, video_id)
        return segments

    def _segments_from_words(self, words, video_id: str, max_gap_ms: int = 1500) -> list[t.Segment]:
        chunks: list[list] = [[]]
        for w in words:
            if chunks[-1] and w.start - chunks[-1][-1].end > max_gap_ms:
                chunks.append([])
            chunks[-1].append(w)
        segments = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            text = " ".join(w.text for w in chunk).strip()
            if not text:
                continue
            segments.append(t.Segment(
                segment_id=f"{video_id}_{i:05d}",
                video_id=video_id,
                start_seconds=chunk[0].start / 1000.0,
                end_seconds=chunk[-1].end / 1000.0,
                transcript_raw=text,
            ))
        return segments
