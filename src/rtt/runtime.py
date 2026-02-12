import os
import shutil
import sys
from pathlib import Path

import httpx

CACHE_DIR = Path(os.environ.get("RTT_CACHE_DIR", Path.home() / ".cache" / "rtt"))
WHISPER_MODEL = "large-v3"
OLLAMA_MODEL = "nomic-embed-text"
OLLAMA_URL = os.environ.get("RTT_OLLAMA_URL", "http://localhost:11434")


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def whisper_cache() -> Path:
    d = cache_dir() / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _check_ollama_model(model: str) -> bool:
    try:
        resp = httpx.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=5)
        return resp.status_code == 200
    except httpx.ConnectError:
        return False


def _check_ollama_running() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except httpx.ConnectError:
        return False


def _check_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _check_assemblyai_key() -> bool:
    return bool(os.environ.get("ASSEMBLYAI_API_KEY"))


def check(
    needs_ffmpeg: bool = False, needs_ollama: bool = False,
    needs_anthropic: bool = False, needs_assemblyai: bool = False,
) -> list[str]:
    errors = []

    if needs_ffmpeg and not _check_binary("ffmpeg"):
        errors.append("ffmpeg not found in PATH — install from https://ffmpeg.org/")

    if needs_ollama:
        if not _check_ollama_running():
            errors.append(f"Ollama not running at {OLLAMA_URL} — start with: ollama serve")
        elif not _check_ollama_model(OLLAMA_MODEL):
            errors.append(f"Ollama model '{OLLAMA_MODEL}' not found — pull with: ollama pull {OLLAMA_MODEL}")

    if needs_anthropic and not _check_anthropic_key():
        errors.append("ANTHROPIC_API_KEY not set — add it to .env or export it")

    if needs_assemblyai and not _check_assemblyai_key():
        errors.append("ASSEMBLYAI_API_KEY not set — add it to .env or export it")

    return errors


def require(
    needs_ffmpeg: bool = False, needs_ollama: bool = False,
    needs_anthropic: bool = False, needs_assemblyai: bool = False,
):
    errors = check(
        needs_ffmpeg=needs_ffmpeg, needs_ollama=needs_ollama,
        needs_anthropic=needs_anthropic, needs_assemblyai=needs_assemblyai,
    )
    if errors:
        print("Missing requirements:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


def ensure_whisper():
    from faster_whisper import WhisperModel
    d = whisper_cache()
    print(f"Loading Whisper {WHISPER_MODEL} (cache: {d})")
    return WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto", download_root=str(d))
