import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

DEFAULT_RTT_DIR = Path(__file__).parent.parent / "data" / "sample"


@pytest.fixture(scope="session")
def server_url():
    rtt_dir = Path(os.environ.get("RTT_DATA_DIR", str(DEFAULT_RTT_DIR)))
    if not rtt_dir.exists():
        pytest.skip(f"{rtt_dir} not found")

    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    proc = subprocess.Popen(
        ["uv", "run", "rtt", "serve", str(rtt_dir), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = f"http://localhost:{port}"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/collections", timeout=1).status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(0.3)
    else:
        proc.terminate()
        proc.wait()
        pytest.fail("Server did not start in time")

    yield url

    proc.terminate()
    proc.wait()


def test_frames_fill_cards(page, server_url):
    page.goto(server_url)
    page.wait_for_selector(".card-thumb img", state="visible", timeout=10_000)
    page.wait_for_load_state("networkidle")

    mismatches = page.evaluate("""() => {
        const imgs = document.querySelectorAll('.card-thumb img');
        const bad = [];
        for (const img of imgs) {
            const ir = img.getBoundingClientRect();
            const cr = img.closest('.card').getBoundingClientRect();
            if (Math.abs(ir.width - cr.width) > 1 || Math.abs(ir.height - cr.height) > 1) {
                bad.push({imgW: ir.width, imgH: ir.height, cardW: cr.width, cardH: cr.height});
            }
        }
        return bad;
    }""")
    assert mismatches == [], f"Images don't fill cards: {mismatches[:3]}"


def test_cards_cover_viewport(page, server_url):
    page.goto(server_url)
    page.wait_for_selector(".card", state="visible", timeout=10_000)
    page.wait_for_load_state("networkidle")

    coverage = page.evaluate("""() => {
        const w = window.innerWidth;
        const h = window.innerHeight;
        let hits = 0, total = 0;
        for (let x = 25; x < w; x += 50) {
            for (let y = 60; y < h; y += 50) {
                total++;
                const el = document.elementFromPoint(x, y);
                if (el && el.closest('.card')) hits++;
            }
        }
        return {hits, total, pct: hits / total};
    }""")
    assert coverage["pct"] > 0.9, f"Only {coverage['pct']:.0%} of viewport covered by cards"


def test_no_placeholder_after_load(page, server_url):
    page.goto(server_url)
    page.wait_for_load_state("networkidle")

    broken = page.evaluate("""() => {
        const cards = document.querySelectorAll('.card');
        const bad = [];
        for (const card of cards) {
            const img = card.querySelector('.card-thumb img');
            const placeholder = card.querySelector('.card-placeholder');
            const text = card.querySelector('.card-text');
            if (placeholder) {
                bad.push({type: 'placeholder', hasText: !!text});
            } else if (img && (!img.complete || img.naturalWidth === 0)) {
                bad.push({type: 'broken-img', src: img.src});
            } else if (!img && !placeholder) {
                bad.push({type: 'empty-thumb', html: card.querySelector('.card-thumb')?.innerHTML?.slice(0, 100)});
            }
        }
        return bad;
    }""")
    assert broken == [], f"{len(broken)} bad cards: {broken[:5]}"


def _largest_black_rect(screenshot_bytes: bytes, threshold: int = 30) -> tuple[int, int]:
    from PIL import Image
    import io
    import numpy as np

    img = np.array(Image.open(io.BytesIO(screenshot_bytes)).convert("L"))
    dark = (img < threshold).astype(np.uint8)
    h, w = dark.shape
    heights = np.zeros(w, dtype=int)
    max_area = 0
    max_dims = (0, 0)
    for row in range(h):
        for col in range(w):
            heights[col] = heights[col] + 1 if dark[row, col] else 0
        stack = []
        for col in range(w + 1):
            cur_h = heights[col] if col < w else 0
            start = col
            while stack and stack[-1][1] > cur_h:
                s, sh = stack.pop()
                area = sh * (col - s)
                if area > max_area:
                    max_area = area
                    max_dims = (col - s, sh)
                start = s
            stack.append((start, cur_h))
    return max_dims


def test_no_gaps_after_filter(page, server_url):
    page.goto(server_url)
    page.wait_for_selector(".card", state="visible", timeout=10_000)

    page.click("#filter-toggle")
    page.click('.filter-only[data-only-collection="tasshin"]')
    page.wait_for_selector(".card", state="visible", timeout=10_000)
    page.wait_for_load_state("networkidle")

    page.click("body", position={"x": 10, "y": 10})
    page.wait_for_timeout(500)

    page.wait_for_timeout(1000)
    shot = page.screenshot()
    bw, bh = _largest_black_rect(shot)
    min_dim = min(bw, bh)
    assert min_dim < 10, f"Largest black rectangle is {bw}x{bh}px — likely a gap in the card grid"
