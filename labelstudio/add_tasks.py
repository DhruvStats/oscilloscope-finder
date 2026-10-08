"""Add new tasks (e.g. captured frames or new photos) to the existing "PoC oscilloscopes" project.

    .venv/Scripts/python labelstudio/make_tasks.py --new raw/captures/2026-10-08 --only-new
    .venv/Scripts/python labelstudio/make_tasks.py --only-new --labelled raw/batch_2026-10-08   (reviewed batch)
    $env:LS_TOKEN = "<your token>"
    .labelstudio-venv/Scripts/python labelstudio/add_tasks.py

The new photos must be under raw/ (the project's photo folder); captured frames already are.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup_project import TITLE, URL, auth_header, call  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    token = os.environ.get("LS_TOKEN")
    if not token:
        sys.exit("set LS_TOKEN to your Label Studio access token first (Account & Settings)")
    path = os.path.join(ROOT, "labelstudio", "tasks_new.json")
    if not os.path.exists(path):
        sys.exit("no labelstudio/tasks_new.json - run make_tasks.py --new <folder> --only-new first")
    tasks = json.load(open(path))
    auth = auth_header(token)
    projects = call("GET", "/api/projects/?page_size=100", auth)
    projects = projects.get("results", projects) if isinstance(projects, dict) else projects
    found = [p for p in projects if p["title"] == TITLE]
    if not found:
        sys.exit(f'project "{TITLE}" not found - create it with setup_project.py')
    pid = found[0]["id"]
    # skip photos that are already in the project (safe to run twice)
    existing, page = set(), 1
    while True:
        r = call("GET", f"/api/tasks/?project={pid}&page={page}&page_size=500", auth)
        batch = r.get("tasks", r) if isinstance(r, dict) else r
        existing |= {t["data"].get("file") for t in batch}
        if len(batch) < 500:
            break
        page += 1
    todo = [t for t in tasks if t["data"]["file"] not in existing]
    if todo:
        call("POST", f"/api/projects/{pid}/import", auth, todo)
    print(f"added {len(todo)} tasks ({len(tasks) - len(todo)} already there): {URL}/projects/{pid}/data")


if __name__ == "__main__":
    main()
