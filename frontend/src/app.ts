interface SearchResult {
  video_id: string;
  segment_id: string;
  start_seconds: number;
  end_seconds: number;
  source_url: string;
  title: string;
  transcript_raw: string;
  transcript_enriched: string;
  frame_url: string | null;
  page_url: string | null;
  score: number;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

async function fetchResults(query: string): Promise<SearchResult[]> {
  const resp = await fetch(`/search?q=${encodeURIComponent(query)}`);
  if (!resp.ok) return [];
  const data: SearchResponse = await resp.json();
  return data.results;
}

function render() {
  const root = document.getElementById("root")!;
  let query = "";
  let results: SearchResult[] = [];
  let activeResult: SearchResult | null = null;

  function update() {
    root.innerHTML = `
      <div class="search-container">
        <h1>Remember That Time</h1>
        <form class="search-bar" id="search-form">
          <input type="text" id="query" placeholder="Search across videos..." value="${esc(query)}" />
          <button type="submit">Search</button>
        </form>
      </div>
      <div id="results" class="results-grid">
        ${results.length === 0 ? '<div class="status">Search for something</div>' : results.map((r, i) => `
          <div class="result-card${r.score > 1.0 ? " weak-match" : ""}" data-idx="${i}">
            ${r.frame_url ? `<img src="${r.frame_url}" alt="" loading="lazy" />` : `<div style="aspect-ratio:16/9;background:#1a1a1a"></div>`}
            <div class="info">
              <div class="title">${esc(r.title)}</div>
              <div class="timestamp">${formatTime(r.start_seconds)} — ${formatTime(r.end_seconds)}</div>
              <div class="transcript">${esc(r.transcript_raw)}</div>
              ${r.score > 1.0 ? '<div class="weak-label">Weak match</div>' : ""}
            </div>
          </div>
        `).join("")}
      </div>
      ${activeResult ? `
        <div class="player-overlay" id="overlay">
          <div class="player-container">
            <button class="close-btn" id="close-player">&times;</button>
            <video id="video-player" controls crossorigin>
              <source src="${esc(activeResult.source_url)}" />
            </video>
            ${activeResult.page_url ? `<a class="source-link" href="${esc(activeResult.page_url)}" target="_blank" rel="noopener">Source</a>` : ""}
          </div>
        </div>
      ` : ""}
    `;

    const input = document.getElementById("query") as HTMLInputElement;
    input.focus();
    input.selectionStart = input.selectionEnd = input.value.length;

    document.getElementById("search-form")!.addEventListener("submit", async (e) => {
      e.preventDefault();
      query = input.value.trim();
      if (!query) return;
      results = await fetchResults(query);
      update();
    });

    document.querySelectorAll(".result-card").forEach((card) => {
      card.addEventListener("click", () => {
        const idx = parseInt((card as HTMLElement).dataset.idx!);
        activeResult = results[idx];
        update();
      });
    });

    if (activeResult) {
      const videoEl = document.getElementById("video-player") as HTMLVideoElement;
      if (videoEl) {
        const seekTo = activeResult.start_seconds;
        videoEl.addEventListener("loadedmetadata", () => { videoEl.currentTime = seekTo; }, { once: true });
        new (window as any).Plyr(videoEl);
      }

      document.getElementById("overlay")!.addEventListener("click", (e) => {
        if ((e.target as HTMLElement).id === "overlay") { activeResult = null; update(); }
      });
      document.getElementById("close-player")!.addEventListener("click", () => {
        activeResult = null; update();
      });
    }
  }

  update();
}

function esc(s: string): string {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

document.addEventListener("DOMContentLoaded", render);
