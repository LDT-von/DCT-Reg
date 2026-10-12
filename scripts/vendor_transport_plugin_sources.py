"""Copy verified model source subsets; never download weights or patient data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


COMMITS = {"mcat": "b9cca63be83c67de7f95308d54a58f80b78b0da1",
           "survpath": "3f73ddd6705ec67d643020c5bb04fb13f9f382cc"}


def vendor(source: Path, destination: Path) -> dict:
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    result = {"schema": "dct_transport_plugin_sources_v1", "repositories": {}}
    plans = []
    for name, commit in COMMITS.items():
        entry = next(e for e in manifest["repositories"] if e["method"].lower() == name)
        if entry["commit"] != commit:
            raise ValueError("unexpected official source commit")
        inventory = json.loads((source / entry["inventory"]).read_text(encoding="utf-8"))
        selected = []
        for item in inventory["files"]:
            relative = Path(item["path"])
            if not ((item["path"].startswith("models/") and relative.suffix == ".py")
                    or relative.name in {"LICENSE", "LICENSE.txt", "README.md", "requirements.txt"}):
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe source inventory path")
            payload = (source / entry["directory"] / relative).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            if (len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]
                    or blob != item["git_blob_sha1"]):
                raise ValueError("source inventory mismatch: " + str(relative))
            selected.append(item)
            plans.append((source / entry["directory"] / relative,
                          destination / entry["directory"] / relative))
        if not any(i["path"].startswith("models/") for i in selected):
            raise ValueError("official model sources are required")
        license_files = [i["path"] for i in selected if Path(i["path"]).name.startswith("LICENSE")]
        result["repositories"][name] = dict(directory=entry["directory"],
            repository_url=entry["repository_url"], commit=commit, files=selected,
            license_files=license_files,
            license_note="Retained upstream license files." if license_files else
                "No standalone license file was present in this pinned upstream snapshot.")
    destination.mkdir(parents=True, exist_ok=False)
    for origin, target in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
    (destination / "SOURCE_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    result = vendor(args.source, args.destination)
    print(json.dumps({"repositories": list(result["repositories"]),
                      "verified_files": sum(len(r["files"]) for r in result["repositories"].values())}))


if __name__ == "__main__":
    main()
