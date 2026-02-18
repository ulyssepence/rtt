declare namespace YT {
  class Player {
    constructor(elementId: string, options: {
      events?: {
        onReady?: (event: { target: Player }) => void;
        onStateChange?: (event: { data: number }) => void;
      };
    });
    getCurrentTime(): number;
    seekTo(seconds: number, allowSeekAhead: boolean): void;
    destroy(): void;
  }
  enum PlayerState { PLAYING = 1 }
}
interface Window { onYouTubeIframeAPIReady?: () => void; }

interface SegmentResult {
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
  collection: string;
  context: string;
  score: number;
}

interface SearchResponse {
  query: string;
  results: SegmentResult[];
}

interface SegmentsResponse {
  segments: SegmentResult[];
  total: number;
  offset: number;
  limit: number;
}

interface CollectionInfo {
  id: string;
  video_count: number;
  segment_count: number;
}

type Mode = "browse" | "search" | "similar";

const PAGE_SIZE = 200;
const CELL_W = 244;
const CELL_H = 140;

let mode: Mode = "browse";
let segments: SegmentResult[] = [];
let searchResults: SegmentResult[] = [];
let activeOverlay: SegmentResult | null = null;
let overlaySegments: SegmentResult[] = [];
let overlayActiveSegment: SegmentResult | null = null;
let query = "";
let browseOffset = 0;
let browseTotal = 0;
let loading = false;
let collections: CollectionInfo[] = [];
let activeCollections: Set<string> = new Set();
let filterOpen = false;
let aboutOpen = false;
let shuffleOrder: number[] = [];
let ytPlayer: YT.Player | null = null;
let ytPollInterval: number | null = null;
let ytApiReady = false;
let ytApiPromise: Promise<void> | null = null;

let panX = 0;
let panY = 0;
let dragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartPanX = 0;
let dragStartPanY = 0;
let dragMoved = false;
let velX = 0;
let velY = 0;
let lastMoveTime = 0;
let lastMoveX = 0;
let lastMoveY = 0;
let inertiaRaf = 0;

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function esc(s: string): string {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function isYouTube(url: string | null): boolean {
  if (!url) return false;
  return url.includes("youtube.com/watch") || url.includes("youtu.be/");
}

function youtubeVideoId(url: string): string | null {
  const m = url.match(/[?&]v=([^&]+)/) || url.match(/youtu\.be\/([^?]+)/);
  return m ? m[1] : null;
}

function loadYTApi(): Promise<void> {
  if (ytApiReady) return Promise.resolve();
  if (ytApiPromise) return ytApiPromise;
  ytApiPromise = new Promise<void>((resolve) => {
    window.onYouTubeIframeAPIReady = () => { ytApiReady = true; resolve(); };
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);
  });
  return ytApiPromise;
}

function updateOverlaySegment(currentTime: number): void {
  if (overlaySegments.length === 0) return;
  let match: SegmentResult | null = null;
  for (const seg of overlaySegments) {
    if (currentTime >= seg.start_seconds && currentTime < seg.end_seconds) { match = seg; break; }
  }
  if (!match) {
    let best = overlaySegments[0];
    let bestDist = Math.abs(currentTime - best.start_seconds);
    for (const seg of overlaySegments) {
      const d = Math.abs(currentTime - seg.start_seconds);
      if (d < bestDist) { best = seg; bestDist = d; }
    }
    match = best;
  }
  if (match && match.segment_id !== overlayActiveSegment?.segment_id) {
    overlayActiveSegment = match;
    const timeEl = document.getElementById("overlay-time");
    const transcriptEl = document.getElementById("overlay-transcript");
    if (timeEl) timeEl.textContent = `${formatTime(match.start_seconds)} — ${formatTime(match.end_seconds)}`;
    if (transcriptEl) transcriptEl.textContent = match.transcript_raw;
  }
}

function collectionParam(): string {
  if (activeCollections.size === 0) return "";
  return `&collections=${[...activeCollections].join(",")}`;
}

async function fetchSegments(offset: number, limit: number): Promise<SegmentsResponse> {
  const resp = await fetch(`/static/segments?offset=${offset}&limit=${limit}${collectionParam()}`);
  return resp.json();
}

async function fetchSearch(q: string, n = 50): Promise<SegmentResult[]> {
  const resp = await fetch(`/search?q=${encodeURIComponent(q)}&n=${n}${collectionParam()}`);
  if (!resp.ok) return [];
  const data: SearchResponse = await resp.json();
  return data.results;
}

async function fetchSimilar(segmentId: string, n = 50): Promise<SegmentResult[]> {
  const resp = await fetch(`/search?segment_id=${encodeURIComponent(segmentId)}&n=${n}${collectionParam()}`);
  if (!resp.ok) return [];
  const data: SearchResponse = await resp.json();
  return data.results;
}

async function fetchVideoSegments(videoId: string): Promise<SegmentResult[]> {
  const resp = await fetch(`/static/video/${encodeURIComponent(videoId)}/segments`);
  if (!resp.ok) return [];
  return resp.json();
}

async function fetchCollections(): Promise<CollectionInfo[]> {
  const resp = await fetch("/collections");
  const data = await resp.json();
  return data.collections;
}

async function loadInitialSegments() {
  loading = true;
  render();
  const data = await fetchSegments(0, PAGE_SIZE);
  segments = data.segments;
  browseTotal = data.total;
  browseOffset = segments.length;
  shuffleOrder = Array.from({ length: segments.length }, (_, i) => i);
  shuffle(shuffleOrder);
  loading = false;
  render();
  centerCanvas();
}

async function loadMoreSegments() {
  if (loading || browseOffset >= browseTotal) return;
  loading = true;
  const data = await fetchSegments(browseOffset, PAGE_SIZE);
  const newStart = segments.length;
  segments.push(...data.segments);
  browseOffset += data.segments.length;
  const newIndices = Array.from({ length: data.segments.length }, (_, i) => newStart + i);
  shuffle(newIndices);
  shuffleOrder.push(...newIndices);
  loading = false;
  render();
}

function shuffle(arr: number[]) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function displaySegments(): SegmentResult[] {
  if (mode === "browse") {
    return shuffleOrder.map(i => segments[i]).filter(s => s.frame_url);
  }
  return searchResults;
}


function renderFilterPanel(): string {
  if (!filterOpen) return "";
  return `
    <div class="filter-panel">
      ${collections.map(c => `
        <label class="filter-item">
          <input type="checkbox" data-collection="${esc(c.id)}" ${activeCollections.has(c.id) ? "checked" : ""} />
          <span>${esc(c.id || "(no collection)")}</span>
          <button class="filter-only" data-only-collection="${esc(c.id)}">only</button>
          <span class="filter-count">${c.video_count} videos, ${c.segment_count} segments</span>
        </label>
      `).join("")}
    </div>
  `;
}

function renderAboutOverlay(): string {
  if (!aboutOpen) return "";
  return `
    <div class="about-overlay" id="about-overlay">
      <div class="about-content">
        <button class="about-close" id="close-about">&times;</button>
        <h2>About RTT</h2>
        <p>RTT (Remember That Time) is a semantic video search engine over thousands of moments from different videos. You can enter search terms at the top and find videos where similar content is present in a video. Try searching “What should I do with my life?” or “alone in the wilderness”</p>
        <p>Browse the collection by dragging the canvas. Tap on the text (large screens) or on the button on each frame to find moments from videos that are similar. Tap on the video to play that part of the video.</p>
        <p>Written by <a target="_blank" href="https://ulyssepence.com">Ulysse Pence</a>.</p>
        <p><a target="_blank" href="https://github.com/ulyssepence/rtt">Source</a></p>
        <p><a target="_blank" href="https://ulyssepence.com/blog/post/remember-that-time-semantic-video-search">Technical write-up</a></p>
      </div>
    </div>
  `;
}

function renderVideoOverlay(): string {
  if (!activeOverlay) return "";
  const seg = activeOverlay;
  const active = overlayActiveSegment || seg;
  const ytId = seg.page_url && isYouTube(seg.page_url) ? youtubeVideoId(seg.page_url) : null;
  const startInt = Math.floor(seg.start_seconds);

  let playerHtml: string;
  if (ytId) {
    playerHtml = `<div class="yt-container"><iframe id="yt-iframe" src="https://www.youtube.com/embed/${ytId}?start=${startInt}&autoplay=1&enablejsapi=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>`;
  } else if (seg.source_url) {
    playerHtml = `<video id="video-player" controls autoplay><source src="${esc(seg.source_url)}" /></video>`;
  } else {
    playerHtml = `<div class="video-loading"><div class="spinner"></div></div>`;
  }

  return `
    <div class="overlay" id="overlay">
      <div class="overlay-content">
        <button class="close-btn" id="close-overlay">&times;</button>
        <div class="overlay-player">${playerHtml}</div>
        <div class="overlay-info">
          ${seg.collection ? `<div class="overlay-collection">${esc(seg.collection)}</div>` : ""}
          <h2 class="overlay-title">${esc(seg.title)}</h2>
          <div class="overlay-actions">
            ${seg.page_url ? `<a class="overlay-action-btn" href="${esc(seg.page_url)}" target="_blank" rel="noopener">SOURCE</a>` : ""}
            <a class="overlay-action-btn" href="#" id="overlay-find-similar">SIMILAR</a>
          </div>
          <div class="overlay-time" id="overlay-time">${formatTime(active.start_seconds)} — ${formatTime(active.end_seconds)}</div>
          <p class="overlay-transcript" id="overlay-transcript">${esc(active.transcript_raw)}</p>
          ${seg.context ? `<p class="overlay-context">${esc(seg.context).replace(/\n/g, '<br>')}</p>` : ""}
        </div>
      </div>
    </div>
  `;
}

interface CellLayout {
  x: number;
  y: number;
  w: number;
  h: number;
}

function layoutCells(items: SegmentResult[], cols: number, isSearch: boolean): CellLayout[] {
  const cells: CellLayout[] = [];
  if (items.length === 0) return cells;

  if (isSearch && cols >= 2) {
    cells.push({ x: 0, y: 0, w: CELL_W * 2, h: CELL_H * 2 });
    const occupied = new Set<string>();
    occupied.add("0,0"); occupied.add("1,0"); occupied.add("0,1"); occupied.add("1,1");
    let idx = 1;
    for (let row = 0; idx < items.length; row++) {
      for (let col = 0; col < cols && idx < items.length; col++) {
        if (!occupied.has(`${col},${row}`)) {
          cells.push({ x: col * CELL_W, y: row * CELL_H, w: CELL_W, h: CELL_H });
          idx++;
        }
      }
    }
  } else {
    for (let i = 0; i < items.length; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      cells.push({ x: col * CELL_W, y: row * CELL_H, w: CELL_W, h: CELL_H });
    }
  }
  return cells;
}

function tileSize(cells: CellLayout[]): { w: number; h: number } {
  let maxX = 0, maxY = 0;
  for (const c of cells) {
    maxX = Math.max(maxX, c.x + c.w);
    maxY = Math.max(maxY, c.y + c.h);
  }
  return { w: maxX, h: maxY };
}

function padItems(items: SegmentResult[], cols: number): SegmentResult[] {
  if (items.length === 0) return items;
  const minCells = cols * 3;
  const target = Math.max(minCells, Math.ceil(items.length / cols) * cols);
  if (items.length >= target) return items;
  const padded: SegmentResult[] = [];
  while (padded.length < target) {
    for (let i = 0; i < items.length && padded.length < target; i++) {
      padded.push(items[i]);
    }
  }
  return padded;
}

function tiledCards(items: SegmentResult[], cols: number, isSearch: boolean): string {
  const originalLen = items.length;
  const wasPadded = items.length < cols * 3;
  items = padItems(items, cols);
  const cells = layoutCells(items, cols, isSearch);
  const tile = tileSize(cells);
  const tilesX = Math.max(3, Math.ceil(window.innerWidth / tile.w) + 2);
  const tilesY = Math.max(3, Math.ceil(window.innerHeight / tile.h) + 2);
  const isMobile = window.innerWidth < 768;

  const cards: string[] = [];
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      for (let i = 0; i < items.length; i++) {
        const c = cells[i];
        const x = tx * tile.w + c.x;
        const y = ty * tile.h + c.y;
        const seg = items[i];
        cards.push(`
          <div class="card" data-idx="${i % originalLen}" style="position:absolute;left:${x}px;top:${y}px;width:${c.w - 2}px;height:${c.h - 2}px;">
            <div class="card-thumb" data-action="play">
              ${seg.frame_url ? `<img src="${seg.frame_url}" alt="" loading="lazy" />` : `<div class="card-placeholder"></div>`}
            </div>
            <div class="card-overlay" data-action="similar" data-segment-id="${seg.segment_id}">
              <span class="card-text">${esc(seg.transcript_raw.slice(0, 120))}${seg.transcript_raw.length > 120 ? "..." : ""}</span>
            </div>
            <button class="similar-btn" data-action="similar" data-segment-id="${seg.segment_id}">✳</button>
          </div>
        `);
      }
    }
  }
  return cards.join("");
}

function render() {
  const root = document.getElementById("root")!;
  const rawItems = displaySegments();
  const isSearch = mode !== "browse";
  const cols = Math.max(1, Math.ceil(Math.sqrt(rawItems.length)));
  const items = padItems(rawItems, cols);
  const cells = layoutCells(items, cols, isSearch);
  const tile = tileSize(cells);
  const tilesX = Math.max(3, Math.ceil(window.innerWidth / tile.w) + 2);
  const tilesY = Math.max(3, Math.ceil(window.innerHeight / tile.h) + 2);
  const totalW = tile.w * tilesX;
  const totalH = tile.h * tilesY;

  root.innerHTML = `
    <div class="top-bar">
      <form class="search-bar" id="search-form">
        <input type="text" id="query" placeholder="Search across videos..." value="${esc(query)}" />
        ${query || isSearch ? `<button type="button" class="clear-btn" id="clear-search">&times;</button>` : ""}
        <button type="submit">Search</button>
        <button type="button" class="filter-btn" id="filter-toggle" title="Filter collections">&#x25A7;</button>
        <button type="button" class="about-btn" id="about-toggle" title="About RTT">?</button>
      </form>
      ${renderFilterPanel()}
    </div>
    <div class="viewport" id="viewport">
      <div class="canvas" id="canvas" style="width:${totalW}px;height:${totalH}px;transform:translate(${panX}px,${panY}px)">
        ${rawItems.length > 0 ? tiledCards(rawItems, cols, isSearch) : ""}
      </div>
      ${loading ? '<div class="loading-indicator"><div class="spinner"></div><div class="loading-text">Searching...</div></div>' : ""}
    </div>
    ${renderAboutOverlay()}
    ${renderVideoOverlay()}
  `;

  bindCanvasDrag();
  bindEvents();
}

function currentTile(): { w: number; h: number } {
  const rawItems = displaySegments();
  if (rawItems.length === 0) return { w: 1, h: 1 };
  const isSearch = mode !== "browse";
  const cols = Math.max(1, Math.ceil(Math.sqrt(rawItems.length)));
  const items = padItems(rawItems, cols);
  return tileSize(layoutCells(items, cols, isSearch));
}

function centerCanvas() {
  const rawItems = displaySegments();
  if (rawItems.length === 0) return;
  const isSearch = mode !== "browse";
  const cols = Math.max(1, Math.ceil(Math.sqrt(rawItems.length)));
  const items = padItems(rawItems, cols);
  const cells = layoutCells(items, cols, isSearch);
  const tile = tileSize(cells);
  const first = cells[0];
  const cx = tile.w + first.x + first.w / 2;
  const cy = tile.h + first.y + first.h / 2;
  panX = -cx + window.innerWidth / 2;
  panY = -cy + window.innerHeight / 2;
  wrapPan();
  updateCanvasTransform();
}

function closeOverlay() {
  if (ytPollInterval != null) { clearInterval(ytPollInterval); ytPollInterval = null; }
  if (ytPlayer) { ytPlayer.destroy(); ytPlayer = null; }
  document.querySelectorAll("video").forEach(v => (v as HTMLVideoElement).pause());
  document.querySelectorAll("iframe").forEach(f => f.remove());
  activeOverlay = null;
  overlaySegments = [];
  overlayActiveSegment = null;
  render();
}

function updateCanvasTransform() {
  const canvas = document.getElementById("canvas");
  if (!canvas) return;
  canvas.style.transform = `translate(${panX}px,${panY}px)`;
}

function wrapPan() {
  const tile = currentTile();
  panX = ((panX % tile.w) + tile.w) % tile.w - tile.w;
  panY = ((panY % tile.h) + tile.h) % tile.h - tile.h;
}

function tickInertia() {
  if (dragging) return;
  if (Math.abs(velX) < 0.5 && Math.abs(velY) < 0.5) return;
  velX *= 0.92;
  velY *= 0.92;
  panX += velX;
  panY += velY;
  wrapPan();
  updateCanvasTransform();
  inertiaRaf = requestAnimationFrame(tickInertia);
}

function setupDragListeners() {
  document.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const now = performance.now();
    const dt = now - lastMoveTime;
    if (dt > 0) {
      velX = (e.clientX - lastMoveX) * (16 / dt);
      velY = (e.clientY - lastMoveY) * (16 / dt);
    }
    lastMoveTime = now;
    lastMoveX = e.clientX;
    lastMoveY = e.clientY;
    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) dragMoved = true;
    panX = dragStartPanX + dx;
    panY = dragStartPanY + dy;
    wrapPan();
    updateCanvasTransform();
  });
  document.addEventListener("pointerup", () => {
    if (!dragging) return;
    dragging = false;
    cancelAnimationFrame(inertiaRaf);
    inertiaRaf = requestAnimationFrame(tickInertia);
  });

  document.addEventListener("wheel", (e) => {
    if ((e.target as HTMLElement).closest(".top-bar")) return;
    if ((e.target as HTMLElement).closest(".overlay")) return;
    e.preventDefault();
    panX -= e.deltaX;
    panY -= e.deltaY;
    wrapPan();
    updateCanvasTransform();
  }, { passive: false });
}

function bindCanvasDrag() {
  const viewport = document.getElementById("viewport");
  if (!viewport) return;

  viewport.addEventListener("pointerdown", (e) => {
    if ((e.target as HTMLElement).closest(".top-bar")) return;
    if ((e.target as HTMLElement).closest(".overlay")) return;
    e.preventDefault();
    cancelAnimationFrame(inertiaRaf);
    velX = 0;
    velY = 0;
    dragging = true;
    dragMoved = false;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartPanX = panX;
    dragStartPanY = panY;
    lastMoveTime = performance.now();
    lastMoveX = e.clientX;
    lastMoveY = e.clientY;
  });
}

function bindEvents() {
  const input = document.getElementById("query") as HTMLInputElement | null;
  if (input && document.activeElement?.tagName !== "INPUT") {
    input.focus();
    input.selectionStart = input.selectionEnd = input.value.length;
  }

  document.getElementById("search-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("query") as HTMLInputElement;
    query = input.value.trim();
    if (!query) return;
    mode = "search";
    loading = true;
    render();
    searchResults = await fetchSearch(query);
    loading = false;
    filterOpen = false;
    render();
    centerCanvas();
  });

  document.getElementById("clear-search")?.addEventListener("click", () => {
    query = "";
    mode = "browse";
    searchResults = [];
    render();
    centerCanvas();
  });

  document.getElementById("filter-toggle")?.addEventListener("click", () => {
    filterOpen = !filterOpen;
    render();
  });

  document.getElementById("about-toggle")?.addEventListener("click", () => {
    aboutOpen = true;
    render();
  });

  document.getElementById("about-overlay")?.addEventListener("pointerdown", (e) => {
    if ((e.target as HTMLElement).id === "about-overlay") { aboutOpen = false; render(); }
  });
  document.getElementById("close-about")?.addEventListener("click", () => {
    aboutOpen = false;
    render();
  });

  document.querySelectorAll(".filter-item input").forEach(cb => {
    cb.addEventListener("change", async (e) => {
      const el = e.target as HTMLInputElement;
      const col = el.dataset.collection!;
      if (el.checked) activeCollections.add(col);
      else activeCollections.delete(col);
      if (mode === "browse") {
        browseOffset = 0;
        await loadInitialSegments();
      } else if (mode === "search" && query) {
        searchResults = await fetchSearch(query);
        render();
      }
    });
  });

  document.querySelectorAll(".filter-only").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      const col = (e.target as HTMLElement).dataset.onlyCollection!;
      if (activeCollections.size === 1 && activeCollections.has(col)) {
        activeCollections = new Set(collections.map(c => c.id));
      } else {
        activeCollections = new Set([col]);
      }
      if (mode === "browse") {
        browseOffset = 0;
        await loadInitialSegments();
      } else if (mode === "search" && query) {
        searchResults = await fetchSearch(query);
        render();
      } else {
        render();
      }
    });
  });

  document.querySelectorAll(".card").forEach(card => {
    card.querySelector("[data-action='play']")?.addEventListener("click", async () => {
      if (dragMoved) return;
      const idx = parseInt((card as HTMLElement).dataset.idx!);
      const items = displaySegments();
      const seg = items[idx];
      const isYT = seg.page_url && isYouTube(seg.page_url);
      activeOverlay = { ...seg, source_url: isYT ? seg.source_url : "" };
      overlayActiveSegment = activeOverlay;
      overlaySegments = [];
      render();
      const [resolved] = await Promise.all([
        isYT ? seg.source_url : fetch(`/video/${encodeURIComponent(seg.video_id)}/resolve`).then(r => r.json()).then(d => d.url as string).catch(() => seg.source_url),
        fetchVideoSegments(seg.video_id).then(s => { overlaySegments = s; }),
      ]);
      if (!activeOverlay) return;
      activeOverlay.source_url = resolved;
      render();
    });

    card.querySelectorAll("[data-action='similar']").forEach(el => {
      el.addEventListener("click", async (e) => {
        if (dragMoved) return;
        e.stopPropagation();
        const segId = (el as HTMLElement).dataset.segmentId!;
        mode = "similar";
        query = "";
        loading = true;
        render();
        searchResults = await fetchSimilar(segId);
        loading = false;
        render();
        centerCanvas();
      });
    });
  });

  if (activeOverlay) {
    const videoEl = document.getElementById("video-player") as HTMLVideoElement | null;
    if (videoEl && activeOverlay) {
      const seekTo = activeOverlay.start_seconds;
      videoEl.addEventListener("loadedmetadata", () => { videoEl.currentTime = seekTo; }, { once: true });
      videoEl.addEventListener("timeupdate", () => updateOverlaySegment(videoEl.currentTime));
    }

    const ytIframe = document.getElementById("yt-iframe");
    if (!videoEl && ytIframe && activeOverlay) {
      const startSeconds = activeOverlay.start_seconds;
      loadYTApi().then(() => {
        ytPlayer = new YT.Player("yt-iframe", {
          events: {
            onReady: (e) => { e.target.seekTo(startSeconds, true); },
            onStateChange: (e) => {
              if (ytPollInterval != null) { clearInterval(ytPollInterval); ytPollInterval = null; }
              if (e.data === YT.PlayerState.PLAYING) {
                ytPollInterval = window.setInterval(() => {
                  if (ytPlayer) updateOverlaySegment(ytPlayer.getCurrentTime());
                }, 500);
              }
            },
          },
        });
      });
    }

    document.getElementById("overlay-find-similar")?.addEventListener("click", async (e) => {
      e.preventDefault();
      if (!overlayActiveSegment) return;
      const segId = overlayActiveSegment.segment_id;
      closeOverlay();
      mode = "similar";
      query = "";
      loading = true;
      render();
      searchResults = await fetchSimilar(segId);
      loading = false;
      render();
      centerCanvas();
    });

    document.getElementById("overlay")?.addEventListener("click", (e) => {
      if (!(e.target as HTMLElement).closest(".overlay-content")) { closeOverlay(); }
    });
    document.getElementById("close-overlay")?.addEventListener("click", closeOverlay);
  }
}

async function init() {
  setupDragListeners();
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (aboutOpen) { aboutOpen = false; render(); }
      else if (activeOverlay) { closeOverlay(); }
      else if (filterOpen) { filterOpen = false; render(); }
    }
  });
  document.addEventListener("pointerdown", (e) => {
    if (!filterOpen) return;
    const panel = document.querySelector(".filter-panel");
    const toggle = document.getElementById("filter-toggle");
    if (panel?.contains(e.target as Node) || toggle?.contains(e.target as Node)) return;
    filterOpen = false;
    render();
  });
  collections = await fetchCollections();
  activeCollections = new Set(collections.map(c => c.id));
  await loadInitialSegments();
}

document.addEventListener("DOMContentLoaded", init);
