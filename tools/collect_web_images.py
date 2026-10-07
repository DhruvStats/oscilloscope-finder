"""Collect candidate product photos from a short list of web pages, with provenance.

For each page: fetch it once, pick product-looking images (og:image, image tags mentioning the model,
dealer gallery links), download those at least --min-side px, skip duplicates, and record the page URL,
image URL, licence note and date in <out>/provenance.csv. Nothing is filtered by model variant here:
every image still needs a human check (TDS2014 vs TDS2014B/C look different) before it enters a dataset.

Sites that block automated access are skipped, never worked around.

    .venv/Scripts/python tools/collect_web_images.py pages.csv raw/web
pages.csv columns: model,url,licence
"""
import csv
import datetime
import hashlib
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) oscilloscope-dataset-research/0.1"
MODEL_WORDS = {"rs_rtb2004": ["rtb2004", "rtb2000", "rtb-2004", "rtb_2004"],
               "tek_tds2014": ["tds2014", "tds-2014", "tds_2014", "tds 2014"],
               "tek_tds1002": ["tds1002", "tds-1002", "tds_1002", "tds 1002"]}


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def candidates(html, base, model):
    words = MODEL_WORDS[model]
    urls = []
    for m in re.finditer(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', html, re.I):
        urls.append(m.group(1))
    for m in re.finditer(r'<(?:img|a|source)[^>]+>', html, re.I):
        tag = m.group(0)
        if not any(w in tag.lower() for w in words + ["product", "gallery", "zoom"]):
            continue
        for attr in ("data-zoom-image", "data-large", "data-src", "href", "src", "srcset"):
            a = re.search(attr + r'=["\']([^"\']+)', tag, re.I)
            if a:
                v = a.group(1).split(",")[-1].strip().split(" ")[0]
                if re.search(r'\.(jpe?g|png|webp)(\?|$)', v, re.I):
                    urls.append(v)
    seen, out = set(), []
    for u in urls:
        u = urllib.parse.urljoin(base, u.replace("&amp;", "&"))
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def main():
    pages, out = sys.argv[1], sys.argv[2]
    min_side = 400
    os.makedirs(out, exist_ok=True)
    prov_path = os.path.join(out, "provenance.csv")
    known = set()
    if os.path.exists(prov_path):
        with open(prov_path, newline="", encoding="utf-8") as f:
            known = {r["sha1"] for r in csv.DictReader(f)}
    new_file = not os.path.exists(prov_path)
    prov = open(prov_path, "a", newline="", encoding="utf-8")
    w = csv.writer(prov)
    if new_file:
        w.writerow(["file", "model", "page_url", "image_url", "licence", "downloaded", "width", "height", "sha1"])
    with open(pages, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        model, url = row["model"], row["url"]
        try:
            body, ctype = fetch(url)
        except Exception as e:  # blocked / gone: skip, do not retry around it
            print(f"skip page {url}: {e}")
            continue
        if ctype.startswith("image/"):
            imgs = [url]
            html = ""
        else:
            html = body.decode("utf-8", "ignore")
            imgs = candidates(html, url, model)
        kept = 0
        for iu in imgs[:30]:
            try:
                data = body if iu == url else fetch(iu)[0]
            except Exception:
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None or min(img.shape[:2]) < min_side:
                continue
            sha = hashlib.sha1(data).hexdigest()
            if sha in known:
                continue
            known.add(sha)
            name = f"{model}_{sha[:10]}.jpg"
            cv2.imwrite(os.path.join(out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            w.writerow([name, model, url, iu, row.get("licence", "unknown - check before use"),
                        datetime.date.today().isoformat(), img.shape[1], img.shape[0], sha])
            kept += 1
            time.sleep(0.5)     # be polite
        print(f"{model:<12} {kept:>2} images  {url}")
        time.sleep(1)
    prov.close()


if __name__ == "__main__":
    main()
