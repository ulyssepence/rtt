from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol, Sequence, Tuple
from urllib import parse as urlparse
import uuid

import httpx
import webvtt
import yt_dlp

from rtt import util


def _extract_video_id(url: str) -> Optional[str]:
    parsed = urlparse.urlparse(url)
    params = urlparse.parse_qs(parsed.query)
    if 'v' in params:
        return params['v'][0]
    if parsed.hostname in ('youtu.be',):
        return parsed.path.lstrip('/')
    return None


def extract_video_id_and_offset(video_url: str, time: str) -> Optional[Tuple[str, Optional[util.Time]]]:
    video_id = _extract_video_id(video_url)
    if not video_id:
        return None

    parsed_url = urlparse.urlparse(video_url)
    query_params = urlparse.parse_qs(parsed_url.query)

    if time:
        offset = util.Time.seconds(int(time))
    elif 't' in query_params or 'time_continue' in query_params:
        time_str = (query_params.get('t') or query_params.get('time_continue'))[0]
        offset = util.Time.seconds(int(time_str))
    else:
        offset = None

    return video_id, offset


def video_url(video_id: str, offset: Optional[util.Time] = None) -> str:
    params = {'v': video_id}
    if offset:
        params['t'] = str(offset.s)
    return f"https://www.youtube.com/watch?{urlparse.urlencode(params)}"


@dataclass(frozen=True)
class Metadata:
    title: str
    author: Optional[str]
    length: util.Time


class Downloader(Protocol):
    def download_video(self, video_id: str, download_dir: Path) -> Path: ...
    def metadata(self, video_id: str) -> Metadata: ...
    def subtitles(
        self,
        video_id: str,
        start: Optional[util.Time] = None,
        stop: Optional[util.Time] = None,
    ) -> Optional[Sequence[str]]: ...


def channel_video_ids(channel_url: str) -> list[dict]:
    if not channel_url.endswith("/videos"):
        channel_url = channel_url.rstrip("/") + "/videos"
    with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        return [{"id": e["id"], "title": e.get("title", "")} for e in info["entries"]]


@dataclass(frozen=True)
class ChannelVideo:
    id: str
    title: str
    description: str
    audio_url: str
    page_url: str

    @property
    def context(self) -> str:
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        return "\n\n".join(parts)


def resolve_video(video_id: str) -> ChannelVideo:
    with yt_dlp.YoutubeDL({"quiet": True, "format": "bestaudio"}) as ydl:
        info = ydl.extract_info(video_url(video_id), download=False)
        return ChannelVideo(
            id=video_id,
            title=info.get("title", ""),
            description=info.get("description", ""),
            audio_url=info["url"],
            page_url=video_url(video_id),
        )


class RealDownloader:

    @classmethod
    def _download(cls, video_id: str, download_dir: Path, fmt: str) -> Path:
        filename = str(uuid.uuid4())
        ydl_opts = {
            'format': fmt,
            'paths': {'home': str(download_dir)},
            'outtmpl': f'{filename}.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url(video_id)])
        for f in download_dir.iterdir():
            if filename in f.name:
                return f
        raise Exception(f'No matching file produced in directory {download_dir}')

    @classmethod
    def download_video(cls, video_id: str, download_dir: Path) -> Path:
        return cls._download(video_id, download_dir, 'bestvideo+bestaudio/best')

    @classmethod
    def download_audio(cls, video_id: str, download_dir: Path) -> Path:
        return cls._download(video_id, download_dir, 'bestaudio')

    @classmethod
    def metadata(cls, video_id: str) -> Metadata:
        info = cls._fetch_info(video_id)

        length_pieces = list(reversed(info['duration_string'].split(':')))
        length = util.Time.zero()
        if 0 < len(length_pieces): length += util.Time.seconds(int(length_pieces[0]))
        if 1 < len(length_pieces): length += util.Time.minutes(int(length_pieces[1]))
        if 2 < len(length_pieces): length += util.Time.hours(  int(length_pieces[2]))

        return Metadata(
            title=info['title'],
            author=info['channel'],
            length=length,
        )

    @classmethod
    def subtitles(
        cls,
        video_id: str,
        start: Optional[util.Time] = None,
        stop: Optional[util.Time] = None,
    ) -> Optional[Sequence[str]]:
        info = cls._fetch_info(video_id)

        def get_subtitles_vtt_url():
            subs = info.get('subtitles', {}).get('en', [])
            match = util.find(lambda s: s.get('ext') == 'vtt', subs)
            return (match or {}).get('url')

        def get_captions_vtt_url():
            captions = info.get('automatic_captions', {}).get('en', [])
            match = util.find(lambda s: s.get('protocol') == 'm3u8_native' and 'url' in s, captions)
            m3u8_url = (match or {}).get('url')
            if not m3u8_url:
                return None

            m3u8 = httpx.get(m3u8_url).text
            return util.find(lambda l: l.startswith('https://www.youtube.com/api/'), m3u8.splitlines())

        vtt_url = get_subtitles_vtt_url() or get_captions_vtt_url()
        if not vtt_url:
            return None

        vtt_text = httpx.get(vtt_url).text
        vtt = webvtt.from_string(vtt_text)

        return list(cls._subtitle_paragraphs(vtt, start, stop))

    @classmethod
    def _subtitle_paragraphs(
        cls,
        vtt: webvtt.WebVTT,
        start: Optional[util.Time],
        stop: Optional[util.Time],
    ) -> Iterator[str]:
        for caption in vtt:
            subtitle_start = cls._parse_vtt_time(caption.start)
            subtitle_stop = cls._parse_vtt_time(caption.end)

            if start and subtitle_stop < start:
                continue
            if stop and stop < subtitle_start:
                break

            for line in caption.text.splitlines():
                yield line

    @classmethod
    def _parse_vtt_time(cls, t: str) -> util.Time:
        dot_pieces = t.split('.')
        time = util.Time.zero() if len(dot_pieces) == 1 else util.Time.millis(int(dot_pieces[1]))

        colon_pieces = list(reversed(dot_pieces[0].split(':')))
        if 0 < len(colon_pieces): time += util.Time.seconds(int(colon_pieces[0]))
        if 1 < len(colon_pieces): time += util.Time.minutes(int(colon_pieces[1]))
        if 2 < len(colon_pieces): time += util.Time.hours(  int(colon_pieces[2]))

        return time

    @classmethod
    def _fetch_info(cls, video_id: str) -> util.Json:
        with yt_dlp.YoutubeDL({'forcejson': True}) as ydl:
            return ydl.extract_info(video_url(video_id), download=False)
