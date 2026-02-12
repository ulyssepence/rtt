import json
import time
import httpx

SEARCH_URL = "https://archive.org/advancedsearch.php"
FIELDS = ["identifier", "title", "description", "date", "creator", "subject"]
QUERY = "collection:prelinger AND mediatype:movies AND -collection:prelingerhomemovies"
ROWS = 500
OUTPUT = "data/prelinger_metadata.json"


def fetch_all():
    client = httpx.Client(timeout=30)
    all_items = []
    page = 1

    first = client.get(SEARCH_URL, params={
        "q": QUERY, "fl[]": ",".join(FIELDS),
        "output": "json", "rows": 1, "page": 1,
    }).json()
    total = first["response"]["numFound"]
    print(f"Total items: {total}")

    while True:
        print(f"  Page {page} ({len(all_items)}/{total})...")
        resp = client.get(SEARCH_URL, params={
            "q": QUERY, "fl[]": ",".join(FIELDS),
            "output": "json", "rows": ROWS, "page": page,
        })
        resp.raise_for_status()
        docs = resp.json()["response"]["docs"]
        if not docs:
            break
        all_items.extend(docs)
        page += 1
        time.sleep(0.5)

    print(f"Fetched {len(all_items)} items")
    return all_items


def to_video_jobs(items: list[dict]) -> list[dict]:
    jobs = []
    for item in items:
        ident = item.get("identifier", "")
        if not ident:
            continue
        title = item.get("title", ident)
        desc = item.get("description", "")
        if isinstance(desc, list):
            desc = " ".join(desc)
        subject = item.get("subject", "")
        if isinstance(subject, list):
            subject = ", ".join(subject)
        context = f"{title}. {desc}" if desc else title
        if subject:
            context += f" Subjects: {subject}"
        source_url = f"https://archive.org/download/{ident}/{ident}_512kb.mp4"

        jobs.append({
            "video_id": ident,
            "title": title,
            "source_url": source_url,
            "context": context,
        })
    return jobs


if __name__ == "__main__":
    items = fetch_all()

    with open(OUTPUT, "w") as f:
        json.dump(to_video_jobs(items), f, indent=2)
    print(f"Wrote {len(items)} video jobs to {OUTPUT}")
