from pathlib import Path
from rtt import youtube

video_id = "R_FQU4KzN7A"
out = Path("output")
out.mkdir(exist_ok=True)

dl = youtube.RealDownloader()

print("Fetching metadata...")
meta = dl.metadata(video_id)
print(f"  {meta.title} by {meta.author} ({meta.length.s}s)")

print("Downloading video...")
path = dl.download_video(video_id, out)
print(f"  Saved to {path}")

print("Fetching subtitles...")
subs = dl.subtitles(video_id)
if subs:
    print(f"  Got {len(subs)} lines")
    for line in subs[:10]:
        print(f"    {line}")
else:
    print("  No subtitles available")
