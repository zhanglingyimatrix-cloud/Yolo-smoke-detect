import argparse
import json
import re
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "videos" / "smoking_samples" / "pexels_smoking.mp4"
DEFAULT_PAGE_URL = "https://www.pexels.com/video/a-man-smoking-cigarette-14500435/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a public smoking MP4 from a Pexels video page.")
    parser.add_argument("--page-url", default=DEFAULT_PAGE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="ignore")


def find_mp4_links(html: str) -> list[str]:
    links = re.findall(r"https://videos\.pexels\.com/video-files/[^\"\\]+?\.mp4", html)
    links = sorted(set(link.replace("\\u002F", "/") for link in links))
    return links


def choose_link(links: list[str]) -> str:
    if not links:
        raise SystemExit("No Pexels MP4 links found on the page.")

    def score(link: str) -> tuple[int, int]:
        match = re.search(r"(\d{3,4})p", link)
        height = int(match.group(1)) if match else 9999
        return (height, len(link))

    return sorted(links, key=score)[0]


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        output.write_bytes(response.read())


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    html = fetch_text(args.page_url)
    links = find_mp4_links(html)
    selected = choose_link(links)
    download(selected, output)

    manifest = {
        "page_url": args.page_url,
        "selected_url": selected,
        "output": str(output),
        "bytes": output.stat().st_size,
        "candidate_count": len(links),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
