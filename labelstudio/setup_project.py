"""Create the "PoC oscilloscopes" project in a running local Label Studio and import the tasks.

Run it yourself once Label Studio is running and you have signed up (locally) and copied your access token
from Account & Settings:

    $env:LS_TOKEN = "<your token>"            # PowerShell
    .labelstudio-venv/Scripts/python labelstudio/setup_project.py

Accepts both a legacy token and a personal access token (refresh token). Talks to http://localhost:8080 only.
"""
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("LS_URL", "http://localhost:8080").rstrip("/")
TITLE = "PoC oscilloscopes"


def call(method, path, auth, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method,
                                 headers={"Authorization": auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def auth_header(token):
    try:
        call("GET", "/api/projects/?page_size=1", f"Token {token}")
        return f"Token {token}"
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
    # personal access token: exchange the refresh token for a short-lived access token
    req = urllib.request.Request(URL + "/api/token/refresh/", data=json.dumps({"refresh": token}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return "Bearer " + json.loads(r.read())["access"]
    except urllib.error.HTTPError as e:
        sys.exit(f"Label Studio rejected the token (HTTP {e.code}). Copy it again from Account & Settings > "
                 "Personal Access Token (or create a new one) and set LS_TOKEN in this same window.")


def main():
    token = os.environ.get("LS_TOKEN")
    if not token:
        sys.exit("set LS_TOKEN to your Label Studio access token first (Account & Settings)")
    auth = auth_header(token)
    config = open(os.path.join(ROOT, "labelstudio", "labeling_config.xml"), encoding="utf-8").read()
    tasks = json.load(open(os.path.join(ROOT, "labelstudio", "tasks.json")))

    existing = call("GET", "/api/projects/?page_size=100", auth)
    existing = existing.get("results", existing) if isinstance(existing, dict) else existing
    found = [p for p in existing if p["title"] == TITLE]
    if found and found[0].get("task_number", 0):
        sys.exit(f'project "{TITLE}" already has tasks: {URL}/projects/{found[0]["id"]}/data')
    if found:   # created by an earlier, interrupted run: finish setting it up
        project = found[0]
        call("PATCH", f"/api/projects/{project['id']}", auth, {"label_config": config})
    else:
        project = None
    project = project or call("POST", "/api/projects/", auth, {
        "title": TITLE, "label_config": config,
        "description": "Bounding boxes for R&S RTB2004, Tektronix TDS 2014, Tektronix TDS 1002. "
                       "See labelstudio/GUIDELINES.md.",
        "show_collab_predictions": True})
    pid = project["id"]
    # let Label Studio serve the photos from raw/ (no copies, no uploads); document root = project folder
    storages = call("GET", f"/api/storages/localfiles/?project={pid}", auth) or []
    if not storages:
        try:
            call("POST", "/api/storages/localfiles/", auth, {
                "project": pid, "title": "lab photos (raw/)", "path": os.path.join(ROOT, "raw"),
                "use_blob_urls": False, "regex_filter": ""})
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")[:400]
            sys.exit(f"Label Studio refused the photo folder: {detail}\n"
                     "Start Label Studio with labelstudio\\start.ps1 (it sets the document root).")
    call("POST", f"/api/projects/{pid}/import", auth, tasks)
    print(f"project {pid} created with {len(tasks)} tasks: {URL}/projects/{pid}/data")


if __name__ == "__main__":
    main()
