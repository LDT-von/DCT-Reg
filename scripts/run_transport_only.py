"""Run an unchanged native trainer with feature-only transport attached.

No patient work happens without invoking this launcher without --dry-run.
Each host runs in its own Python process / upstream environment. The launcher
never imports a DCT objective or replaces the host's loss/forward implementation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
HOSTS = {
    "slotspe": ("models.SlotSPE", "SlotSPE", "survival.py", None),
    "survpath": ("models.model_SurvPath", "SurvPath", "main.py", None),
    "pibd": ("models.model_PIBD", "PIBD", "main.py", None),
    "mmp": ("mil_models.model_multimodal", "coattn", None, "training.main_survival"),
    "ld_cvae": ("models.model_coattn", "Robust_MCAT", "main.py", None),
    "tta": ("mil_models.model_otmil_label", "OTMIL_Label", None, "training.main_survival"),
}


def _argument_value(arguments, flag, default):
    value = default
    for index, item in enumerate(arguments):
        if item == flag and index + 1 < len(arguments):
            value = arguments[index + 1]
        elif item.startswith(flag + "="):
            value = item.split("=", 1)[1]
    return value


def _validate_native_selection(name, arguments):
    if name == "mmp" and _argument_value(arguments, "--model_mm_type", "coattn") != "coattn":
        raise ValueError("MMP transport comparison requires the native --model_mm_type coattn arm")
    if name == "ld_cvae" and _argument_value(arguments, "--model_type", "mcat") != "mcat":
        raise ValueError("LD-CVAE currently supports --model_type mcat (Robust_MCAT)")
    if name == "tta" and (
        _argument_value(arguments, "--modality_type", "gene") != "multi" or
        _argument_value(arguments, "--fusion_type", "coattn") != "coattn"
    ):
        raise ValueError("TTA requires --modality_type multi and --fusion_type coattn")


@contextmanager
def _native_import_compatibility(name):
    # Upstream LD-CVAE imports the pre-PyTorch-2 name for the same Linear
    # implementation. Scope this name alias to import; no source/math changes.
    import torch.nn.modules.linear as linear
    temporary = name == "ld_cvae" and not hasattr(linear, "_LinearWithBias")
    if temporary:
        linear._LinearWithBias = linear.NonDynamicallyQuantizableLinear
    try:
        yield
    finally:
        if temporary:
            del linear._LinearWithBias


def make_plan(host, arm, source_root, native_args, mechanism_config=None):
    """Pure metadata generation: no upstream imports, tensors or patient reads."""
    name = host.lower().replace("-", "_")
    if name not in HOSTS or arm not in {"native", "bypass", "independent", "learned"}:
        raise ValueError("unsupported host or comparison arm")
    _validate_native_selection(name, native_args)
    source = Path(source_root).resolve()
    module, class_name, entry_file, entry_module = HOSTS[name]
    prefix = "src/" if name in {"mmp", "tta"} else ""
    model_file = source / prefix / (module.replace(".", "/") + ".py")
    entry = source / (entry_file or prefix + entry_module.replace(".", "/") + ".py")
    if not model_file.is_file() or not entry.is_file():
        raise FileNotFoundError(f"complete native source is required: {model_file}, {entry}")
    values = {"transport_mode": "balanced", "eps": .05, "eps_start": .05,
              **dict(mechanism_config or {})}
    if "plan_mode" in values:
        raise ValueError("plan_mode belongs to --arm; remove it from mechanism config")
    # Validate keys, including rejection of old loss/reconstruction fields.
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from survot_rank.plugins.transport_only.mechanism import TransportOnlyConfig
    TransportOnlyConfig(**values)
    if values.get("eps", .05) != values.get("eps_start", .10):
        raise ValueError("native launcher uses fixed epsilon; use eps_start=eps or explicit manual set_epoch integration")
    return {
        "schema": "dct_transport_only_native_run_v1", "host": name, "arm": arm,
        "source_root": str(source), "model_module": module, "model_class": class_name,
        "model_file": str(model_file),
        "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "entry_file": entry_file, "entry_module": entry_module,
        "native_args": list(native_args), "mechanism_config": values,
        "loss_owner": "native_host_only", "dct_auxiliary_losses": [],
        "patient_training_executed": False,
    }


def run_native(plan, record):
    """Explicit execution path; native command-line/data handling is unchanged."""
    record = Path(record).resolve()
    record.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a new experiment overwriting a previous record.
    with record.open("x", encoding="utf-8") as stream:
        json.dump({**plan, "status": "starting"}, stream, indent=2, ensure_ascii=False)
    source = Path(plan["source_root"])
    native_search = source / "src" if plan["host"] in {"mmp", "tta"} else source
    sys.path.insert(0, str(native_search))
    previous_cwd, previous_argv = Path.cwd(), sys.argv[:]
    previous_init, model_class = None, None
    attachments = []
    native_instances = 0
    def record_status(status, error=None):
        payload = {**plan, "status": status, "native_models_constructed": native_instances,
                   "patient_training_executed": None,
                   "execution_note": "native entry invoked; completion/metrics require native outputs"}
        payload["attachments"] = [{"target": item.target_name,
                                    "config": item.mechanism.config.to_dict(),
                                    "diagnostics": item.mechanism.last_diagnostics}
                                   for item in attachments]
        if error:
            payload["error"] = error
        record.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        os.chdir(source)
        from survot_rank.plugins.transport_only.hosts import attach_transport
        with _native_import_compatibility(plan["host"]):
            model_class = getattr(importlib.import_module(plan["model_module"]), plan["model_class"])
        previous_init = model_class.__init__
        @wraps(previous_init)
        def initialize(model, *args, **kwargs):
            nonlocal native_instances
            previous_init(model, *args, **kwargs)
            native_instances += 1
            if plan["arm"] != "native":
                item = attach_transport(model, plan["host"],
                    {**plan["mechanism_config"],
                     "plan_mode": "independent" if plan["arm"] == "independent" else "learned"},
                    enabled=plan["arm"] != "bypass")
                attachments.append(item)
        model_class.__init__ = initialize
        sys.argv = [plan["entry_file"] or plan["entry_module"], *plan["native_args"]]
        if plan["entry_module"]:
            runpy.run_module(plan["entry_module"], run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(str(source / plan["entry_file"]), run_name="__main__")
        if plan["arm"] != "native" and not attachments:
            raise RuntimeError("native trainer did not construct the requested host class; check native model flags")
        if plan["arm"] in {"learned", "independent"} and not any(
            item.mechanism.last_diagnostics.get("enabled") for item in attachments
        ):
            raise RuntimeError("native trainer never used the selected transport insertion point")
        record_status("native_entry_completed")
    except BaseException as error:
        # argparse --help / intentional zero exit is not experiment completion.
        record_status("native_entry_failed_or_exited", f"{type(error).__name__}: {error}")
        raise
    finally:
        if previous_init is not None:
            model_class.__init__ = previous_init
        os.chdir(previous_cwd)
        sys.argv = previous_argv
        sys.path.remove(str(native_search))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=HOSTS, default="slotspe")
    parser.add_argument("--arm", choices=("native", "bypass", "independent", "learned"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mechanism-config", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    native_args = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
    values = json.loads(args.mechanism_config.read_text(encoding="utf-8-sig")) if args.mechanism_config else None
    plan = make_plan(args.host, args.arm, args.source_root, native_args, values)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if args.record is None:
        parser.error("--record is required for execution; use a new record and a distinct native output directory per arm")
    run_native(plan, args.record)


if __name__ == "__main__":
    main()
