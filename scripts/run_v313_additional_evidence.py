#!/usr/bin/env python3
"""v3.13 reconstruction specificity, patch deletion and patch-budget diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Dry-run by default; --execute loads real checkpoint/data")
    run.add_argument("--manifest", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--experiment", choices=("all", "reconstruction", "patches"), default="all")
    run.add_argument("--arm", action="append")
    run.add_argument("--cancer", action="append")
    run.add_argument("--fold", action="append", type=int, choices=range(5))
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--fractions", default="0,0.1,0.25,0.5,0.75")
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--diagnostic-seed", type=int, default=20261008)
    run.add_argument("--execute", action="store_true")
    plot = commands.add_parser("plot", help="PDF/PNG plots from completed hashed exports")
    plot.add_argument("--exports", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plot":
            from survot_rank.evidence.additional_plots import make_additional_figures
            index = make_additional_figures(args.exports, args.output)
            print(f"[additional] {len(index)} figures (PDF + PNG)")
            return
        from survot_rank.evidence.manifest import audit, load_manifest
        from survot_rank.evidence.additional import export_additional, validate_options
        fractions = validate_options(args.experiment, args.fractions.split(","), args.repeats)
        manifest = load_manifest(args.manifest)
        selected = [r for r in manifest["runs"] if (not args.arm or r["arm"] in args.arm)
            and (not args.cancer or r["cancer"] in args.cancer)
            and (args.fold is None or r["fold"] in args.fold)]
        if not selected:
            raise ValueError("No matching runs")
        print(json.dumps(dict(execute=args.execute, experiment=args.experiment,
            runs=[r["id"] for r in selected], fractions=fractions, repeats=args.repeats,
            diagnostic_seed=args.diagnostic_seed, output=args.output), indent=2))
        if not args.execute:
            print("[additional] dry-run; no checkpoint/data loaded, no inference, no output written")
            return
        report = audit(manifest)
        if not report["passed"]:
            raise ValueError("; ".join(report["errors"]))
        # Check the entire destination set before starting any real inference.
        for r in selected:
            if Path(r["id"]).name != r["id"] or r["id"] in (".", "..") or not r.get("checkpoint"):
                raise ValueError("Require simple run IDs and best checkpoints")
            out = Path(args.output) / r["id"]
            if out.exists() and any(out.iterdir()):
                raise ValueError(f"Refuse to overwrite {out}; use fresh output")
        for r in selected:
            export_additional(r, args.output, experiment=args.experiment, device=args.device,
                fractions=fractions, repeats=args.repeats, diagnostic_seed=args.diagnostic_seed)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"[additional] {error}\n")


if __name__ == "__main__":
    main()
