# RTT Frontend Redesign Spec

## Context

The current frontend is a simple search-first UI: blank page with a search bar, results appear as a grid of cards. This redesign makes the interface browsable by default — a wall of thumbnails you can explore before searching. Search becomes a lens that reorganizes the space rather than populating an empty page.

## Core Interaction Model

### Default State (No Query)

- Full-viewport grid of all segment thumbnails, randomly shuffled on each page load
- Search bar floats at top center, semi-transparent background
- Each thumbnail has a semi-transparent overlay at the bottom showing a 1-2 line excerpt of the transcript

### Search

- On search, the grid reorganizes: best match moves to center, surrounding thumbnails are the next-closest results in vector space
- Layout: ranked spiral — center = rank 1, ring around it = ranks 2-8, next ring = 9-20, etc. Not true dimensionality reduction, just ranked proximity
- Scrolling outward loads progressively weaker matches
- Clearing the search returns to the default shuffled layout

### Clicking a Transcript Overlay (Desktop)

- Uses that segment's embedding as a query
- Grid rearranges: clicked segment moves to center, surrounding thumbnails show nearest neighbors in vector space
- Effectively a "find similar" action without typing

### Clicking a Transcript "Find Similar" Icon (Mobile)

- Same behavior as clicking transcript on desktop, but triggered via a small icon button on the card (since two click zones are too small on mobile)

### Clicking a Thumbnail (Outside Transcript)

- Opens a video overlay (like desktop Instagram):
  - Desktop: Large video player on the left (~65% width), info panel on the right (~35%)
  - Mobile: Video at top, info panel below
- Info panel contains: title, source link (YouTube page / Internet Archive page / local path), description/context
- Video starts playing automatically from the segment's timestamp
- YouTube videos: embed YouTube iframe player with `?start=X&autoplay=1`
- Remote non-YouTube: standard HTML5 video player with Plyr
- Local videos: served via existing `/video/{id}` route

### Filtering by Source

- Filter button next to search bar
- Opens a dropdown/overlay listing all available collections (YouTube channels, IA collections, local directories)
- Each collection is a toggle (on/off)
- Filtering constrains both the default grid and search results
- Collections are derived from a new `collection` field on each video

## Data Model Changes

### New `collection` field

Add to `Video` dataclass, `.rtt` manifest, LanceDB schema, and segment rows.

Files to modify:
- `src/rtt/types.py` — add `collection: str = ""` to `Video` and `VideoJob`
- `src/rtt/package.py` — include `collection` in manifest write/read
- `src/rtt/vector.py` — add `collection` to SCHEMA
- `src/rtt/server.py` — pass collection through, add filter param to search, add `/collections` endpoint
- `src/rtt/batch.py` — propagate collection from job to video
- `src/rtt/__main__.py` — accept `--collection` flag, auto-derive from YouTube channel name or IA collection URL

### Search API Changes

- `GET /search?q=...&collections=id1,id2` — filter by collection
- `GET /search?segment_id=...` — "find similar" by segment (vector lookup by ID instead of text query)
- `GET /collections` — list all loaded collections with video counts
- `GET /segments?offset=0&limit=100&collections=...` — paginated segment listing for the default grid

## Frontend Architecture

### Technology

- Keep the current vanilla TS + esbuild setup (no framework)
- Add a masonry layout library or implement CSS-only masonry (`columns` property or CSS `masonry` if targeting modern browsers)
- YouTube iframe API for YouTube embeds

### Layout

- CSS masonry grid for thumbnails, full viewport
- Fixed-position search bar at top
- Overlay for video player (existing pattern, extended with info panel)
- Overlay for filter panel

### Responsive

- Desktop (>768px): Masonry grid, video overlay is side-by-side (video left, info right)
- Mobile (<768px): Single-column or 2-column grid, video overlay is stacked (video top, info bottom), search bar at top. Don't attempt spatial proximity metaphor — just show ranked results in order.

## Design Decisions

1. Click UX (desktop): Click transcript overlay = "find similar" (vector-proximity rearrangement). Click thumbnail (outside transcript) = open video. Two distinct zones on the card.
2. Click UX (mobile): Cards show a small "find similar" icon button instead of the two-zone approach (too small on mobile). Tap card = open video, tap icon = find similar.
3. Default order: Random shuffle on each page load. Encourages exploration.
4. Masonry vs uniform grid: Thumbnails are all 16:9 but transcript overlays vary height slightly — CSS columns or uniform grid with variable text is fine. True masonry unnecessary.
5. Autoplay concerns: YouTube iframe autoplay is restricted in many browsers without user interaction. May need a "click to play" fallback.
6. Initial load performance: Thousands of segments on page load is heavy. Need paginated `/segments` endpoint + infinite scroll (load ~50 at a time).

## Verification

- Load server with multiple .rtt files from different sources
- Verify masonry grid renders on page load with thumbnails and transcript overlays
- Search and confirm spatial reorganization (center = best match)
- Click transcript → verify "find similar" rearranges grid
- Click thumbnail → verify video overlay with info panel
- Test YouTube embed with timestamp autoplay
- Filter by collection and verify results narrow
- Test on mobile viewport
