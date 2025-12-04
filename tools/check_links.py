#!/usr/bin/env python3
"""
Link Checker for Hugo Website
Checks all internal links and download links on the deployed site.
"""

import requests
from urllib.parse import urljoin
import re
import time

BASE_URL = "https://digital-ai-finance.github.io/ML_Design_Thinking_16/"
BASE_PATH = "/ML_Design_Thinking_16"
HOST = "https://digital-ai-finance.github.io"

def get_all_links(url, session):
    """Extract all links from a page."""
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return [], resp.status_code

        html = resp.text
        links = []

        # Extract href and src links
        patterns = [
            r'href="([^"]+)"',
            r"href='([^']+)'",
            r'href=([^\s>]+)',
            r'src="([^"]+)"',
            r"src='([^']+)'",
            r'src=([^\s>]+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, html):
                link = match.group(1).strip()
                if link.startswith('mailto:') or link.startswith('javascript:') or link.startswith('#'):
                    continue

                # Build full URL
                if link.startswith('http'):
                    full_url = link
                elif link.startswith('/'):
                    full_url = HOST + link
                else:
                    full_url = urljoin(url, link)

                links.append(full_url)

        return list(set(links)), 200
    except Exception as e:
        return [], str(e)

def check_link(url, session):
    """Check if a link is accessible."""
    try:
        resp = session.head(url, timeout=15, allow_redirects=True)
        if resp.status_code == 405:
            resp = session.get(url, timeout=15, stream=True)
        return resp.status_code
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except requests.exceptions.ConnectionError:
        return "CONN_ERR"
    except Exception as e:
        return str(e)[:20]

def is_our_site(url):
    """Check if URL is on our Hugo site (not GitHub or external)."""
    return url.startswith(HOST + BASE_PATH)

def is_download(url):
    """Check if URL is a download link."""
    return "/downloads/" in url and url.endswith(".pdf")

def is_image(url):
    """Check if URL is an image."""
    return any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'])

def main():
    print("=" * 60)
    print("Link Checker for ML Design Thinking Website")
    print("=" * 60)
    print(f"\nBase URL: {BASE_URL}\n")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) LinkChecker/1.0'
    })

    visited = set()
    to_visit = [BASE_URL]
    all_links = {}
    download_links = set()
    image_links = set()

    print("Crawling site...")

    while to_visit:
        url = to_visit.pop(0)
        url = url.rstrip('/')

        if url in visited:
            continue

        # Only crawl our Hugo site pages
        if not is_our_site(url):
            continue
        if any(url.lower().endswith(ext) for ext in ['.pdf', '.png', '.jpg', '.css', '.js', '.svg']):
            continue

        visited.add(url)
        short = url.replace(HOST + BASE_PATH, "") or "/"
        print(f"  {short}")

        links, status = get_all_links(url, session)

        if status != 200:
            print(f"    ERROR: {status}")
            continue

        for link in links:
            link = link.rstrip('/')

            # Only process our site links
            if not is_our_site(link):
                continue

            if link not in all_links:
                all_links[link] = url

            if is_download(link):
                download_links.add(link)
            elif is_image(link):
                image_links.add(link)

            if link not in visited:
                if not any(link.lower().endswith(ext) for ext in ['.pdf', '.png', '.jpg', '.css', '.js', '.svg']):
                    to_visit.append(link)

        time.sleep(0.1)

    print(f"\nCrawled {len(visited)} pages")
    print(f"Found {len(download_links)} downloads, {len(image_links)} images")

    # Check downloads
    print("\n" + "-" * 60)
    print("DOWNLOADS")
    print("-" * 60)

    dl_ok, dl_fail = 0, 0
    for url in sorted(download_links):
        status = check_link(url, session)
        fname = url.split("/")[-1]
        if status == 200:
            print(f"  [OK]   {fname}")
            dl_ok += 1
        else:
            print(f"  [FAIL] {fname} ({status})")
            dl_fail += 1
        time.sleep(0.05)

    # Check images
    print("\n" + "-" * 60)
    print("IMAGES")
    print("-" * 60)

    img_ok, img_fail = 0, 0
    for url in sorted(image_links):
        status = check_link(url, session)
        fname = url.split("/")[-1]
        if status == 200:
            print(f"  [OK]   {fname}")
            img_ok += 1
        else:
            print(f"  [FAIL] {fname} ({status})")
            img_fail += 1
        time.sleep(0.05)

    # Check internal pages
    print("\n" + "-" * 60)
    print("PAGES")
    print("-" * 60)

    pg_ok, pg_fail = 0, 0
    internal = [u for u in all_links if is_our_site(u) and not is_download(u) and not is_image(u)]
    for url in sorted(set(internal)):
        if any(url.lower().endswith(ext) for ext in ['.css', '.js']):
            continue
        status = check_link(url, session)
        short = url.replace(HOST + BASE_PATH, "") or "/"
        if status == 200:
            print(f"  [OK]   {short}")
            pg_ok += 1
        else:
            print(f"  [FAIL] {short} ({status})")
            pg_fail += 1
        time.sleep(0.05)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pages crawled: {len(visited)}")
    print(f"Downloads: {dl_ok} OK, {dl_fail} FAIL")
    print(f"Images: {img_ok} OK, {img_fail} FAIL")
    print(f"Pages: {pg_ok} OK, {pg_fail} FAIL")

    total_fail = dl_fail + img_fail + pg_fail
    print("\n" + "=" * 60)
    if total_fail == 0:
        print("ALL LINKS OK!")
    else:
        print(f"FAILED: {total_fail} broken links")
    print("=" * 60)

    return 1 if total_fail > 0 else 0

if __name__ == "__main__":
    exit(main())
