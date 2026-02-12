import asyncio
import subprocess
from pathlib import Path

REMOTE_CONCURRENCY = 20


def extract(video_path: Path, timestamps: list[float], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(timestamps)
    paths = []
    for idx, ts in enumerate(timestamps):
        out = output_dir / f"{int(ts):06d}.jpg"
        result = subprocess.run(
            ["ffmpeg", "-ss", str(ts), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", "-y", str(out)],
            capture_output=True,
        )
        if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            out.unlink(missing_ok=True)
            paths.append(None)
        else:
            paths.append(out)
        print(f"\r  Extracting frames: {idx + 1}/{total}", end="", flush=True)
    if total > 0:
        print()
    return paths


async def extract_remote(
    source_url: str, timestamps: list[float], output_dir: Path,
    concurrency: int = REMOTE_CONCURRENCY,
) -> list[Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    total = len(timestamps)
    done = 0

    async def _one(ts: float) -> Path | None:
        nonlocal done
        out = output_dir / f"{int(ts):06d}.jpg"
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-ss", str(ts), "-i", source_url,
                "-frames:v", "1", "-q:v", "2", "-y", str(out),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        done += 1
        print(f"\r  Extracting remote frames: {done}/{total}", end="", flush=True)
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            out.unlink(missing_ok=True)
            return None
        return out

    tasks = [_one(ts) for ts in timestamps]
    results = await asyncio.gather(*tasks)
    if total > 0:
        print()
    return list(results)
