"""Check or install the scoped v23 package without starting a patient job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess


def within(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Invalid bundle-relative path")
    result = root.joinpath(*part.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Path escapes the intended project")
    return result


def check_patch(project, patch, reverse=False):
    command = ["git", "apply", "--ignore-space-change", "--check"]
    if reverse:
        command.append("--reverse")
    command.append(str(patch))
    return subprocess.run(command, cwd=project, capture_output=True, text=True)


def same_file_contents(existing: bytes, incoming: bytes) -> bool:
    """Treat Windows CRLF and Unix LF copies of the same UTF-8 text as equal."""
    if existing == incoming:
        return True
    try:
        left = existing.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        right = incoming.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        return False
    return left == right


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check only, the default")
    args = parser.parse_args(argv)
    if args.apply and args.check:
        parser.error("Choose either check or apply")
    project = args.project.resolve(strict=True)
    bundle = args.bundle.resolve(strict=True)
    if not (project / "survot_rank/research/methods/dct_v21/model.py").is_file():
        raise ValueError("An existing independent DCT v2.1 checkout is required")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["payload"]:
        source = within(bundle, entry["bundle_path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Bundle integrity failed: " + entry["bundle_path"])
    additions = []
    for relative in manifest["new_files"]:
        source, target = within(bundle, "new_files/" + relative), within(project, relative)
        if target.exists() and not same_file_contents(target.read_bytes(), source.read_bytes()):
            raise FileExistsError("Existing material preserved: " + str(target))
        additions.append((source, target))
    patch_actions = []
    for entry in manifest["shared_files"]:
        within(project, entry["target"])
        selected = None
        already_applied = False
        failures = []
        for relative in entry["patches"]:
            patch = within(bundle, relative)
            result = check_patch(project, patch)
            if result.returncode == 0:
                selected = patch
                break
            if check_patch(project, patch, reverse=True).returncode == 0:
                selected, already_applied = patch, True
                break
            failures.append(result.stderr)
        if selected is None:
            raise RuntimeError("Integration conflict, no files changed: " + entry["target"] + "\n" + "\n".join(failures))
        patch_actions.append((entry["target"], selected, already_applied))
    agents = within(project, "AGENTS.md")
    prior = agents.read_text(encoding="utf-8") if agents.exists() else ""
    sections = []
    for relative in manifest["instruction_sections"]:
        section = within(bundle, relative).read_text(encoding="utf-8").strip()
        marker = section.splitlines()[0]
        if marker in prior and section not in prior:
            raise RuntimeError("Existing version instructions differ, no files changed: " + marker)
        if marker not in prior:
            sections.append(section)
    report = dict(project=str(project), patient_jobs_started=False,
                  new_files_needed=sum(not target.exists() for _, target in additions),
                  integration_already_applied=all(done for _, _, done in patch_actions))
    if not args.apply:
        report["status"] = "check_passed_no_changes"
        print(json.dumps(report, ensure_ascii=False))
        return report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup = within(project, "docs/history/dct_v23_install_" + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    for relative in [entry[0] for entry in patch_actions] + (["AGENTS.md"] if agents.exists() else []):
        original = within(project, relative)
        if original.exists():
            target = within(backup, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
    for _, patch, done in patch_actions:
        if not done:
            subprocess.run(["git", "apply", "--ignore-space-change", str(patch)], cwd=project, check=True)
    for source, target in additions:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if sections:
        agents.write_text(prior.rstrip() + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    report.update(status="installed", backup=str(backup))
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
