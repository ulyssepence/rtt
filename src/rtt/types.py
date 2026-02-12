from dataclasses import dataclass, field


@dataclass
class Segment:
    segment_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    transcript_raw: str
    transcript_enriched: str = ""
    text_embedding: list[float] = field(default_factory=list)
    frame_path: str = ""
    has_speech: bool = True
    source: str = "transcript"


@dataclass
class Video:
    video_id: str
    title: str
    source_url: str
    context: str
    duration_seconds: float
    status: str = "new"


@dataclass
class VideoJob:
    video_id: str
    title: str
    source_url: str
    context: str = ""
