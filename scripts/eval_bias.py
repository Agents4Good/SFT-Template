"""Validate measured pre-flight sizing against post-run telemetry.

Reads every src/output/*/run-*/peak_memory.json, preferring the embedded
`preflight` report (new real-Trainer probes), then a sibling
preflight.json, then the legacy `micro_search` calibration schema.

Per run it reports:
- selected micro / effective batch and the measured limit
- probe peak vs actual sustained peak (gap)
- margin left below the limit and the safety verdict

Usage:
    .venv/bin/python scripts/eval_bias.py [--output-dir src/output] [--strict]

--strict exits 2 if any run overran its usable memory or its limit, or
if a legacy calibration's predicted gap was off by > 1.5 GB.
"""

import argparse
import glob
import json
import os
import sys

LEGACY_HEADROOM = 5.0
TOLERANCE_GB = 1.5
TIGHT_GB = 0.5
CONSERVATIVE_GB = 3.0


def load_runs(output_dir):
    paths = sorted(glob.glob(os.path.join(output_dir, "*", "peak_memory.json")))
    runs = []
    for p in paths:
        try:
            with open(p) as f:
                runs.append((p, json.load(f)))
        except (OSError, ValueError) as exc:
            print(f"WARN: skipping {p}: {exc}", file=sys.stderr)
    return runs


def _load_preflight(path, data):
    pf = data.get("preflight")
    if isinstance(pf, dict):
        return pf
    sibling = os.path.join(os.path.dirname(path), "preflight.json")
    if os.path.exists(sibling):
        try:
            with open(sibling) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    return None


def _selected_probe_peak(preflight):
    micro = preflight.get("micro", preflight.get("selected_micro"))
    cands = [c for c in (preflight.get("candidates") or [])
             if c.get("peak_reserved_gb") is not None
             and c.get("micro") == micro]
    confirm = [c for c in cands if c.get("result") == "confirm-pass"]
    if confirm:
        return confirm[-1]["peak_reserved_gb"]
    passes = [c for c in cands if c.get("result") in ("pass", "confirm-pass")]
    if passes:
        return passes[-1]["peak_reserved_gb"]
    check = preflight.get("check") or {}
    return check.get("peak_reserved_gb")


def _new_verdict(actual_res, usable, limit):
    if actual_res is None:
        return "DRY-RUN"
    if usable is not None and actual_res >= usable:
        return "OVER-USABLE"
    if limit is not None and actual_res > limit:
        return "OVER-LIMIT"
    if limit is not None and limit - actual_res < TIGHT_GB:
        return "TIGHT"
    if limit is not None and limit - actual_res > CONSERVATIVE_GB:
        return "CONSERVATIVE"
    return "SAFE"


def _legacy(data):
    ms = data.get("micro_search") or {}
    cal = ms.get("calibration") or {}
    total = cal.get("device_total_gb") or 12.0
    actual_res = data.get("actual_peak_reserved_gb")
    actual_alc = data.get("actual_peak_allocated_gb")
    bias = cal.get("bias_gb")
    probe_peak = cal.get("selected_probe_peak_gb")
    gap = (actual_res - probe_peak
           if actual_res is not None and probe_peak is not None else None)
    bias_err = gap - bias if gap is not None and bias is not None else None
    free = total - actual_res if actual_res is not None else None

    if actual_res is not None and actual_res >= total:
        verdict = "OOM/OVER"
    elif bias_err is not None and abs(bias_err) > TOLERANCE_GB:
        verdict = "BIAS-OFF"
    elif free is not None and free < 0.75:
        verdict = "TIGHT"
    elif free is not None and free > CONSERVATIVE_GB:
        verdict = "CONSERVATIVE"
    elif actual_res is not None:
        verdict = "SAFE"
    else:
        verdict = "DRY-RUN"

    return {
        "run": None,
        "mode": "legacy",
        "micro": data.get("per_device_train_batch_size"),
        "effective": ms.get("effective_batch_size"),
        "limit": None,
        "probe_peak": probe_peak,
        "actual_res": actual_res,
        "actual_alc": actual_alc,
        "gap": gap,
        "margin": free,
        "bias_err": bias_err,
        "verdict": verdict,
    }


def evaluate(path, data):
    pf = _load_preflight(path, data)
    actual_res = data.get("actual_peak_reserved_gb")
    actual_alc = data.get("actual_peak_allocated_gb")

    if not pf:
        row = _legacy(data)
        row["run"] = os.path.basename(os.path.dirname(path))
        return row

    micro = pf.get("micro", pf.get("selected_micro"))
    if micro is None:
        micro = data.get("per_device_train_batch_size")
    accum = pf.get("accum", pf.get("gradient_accumulation_steps"))
    effective = pf.get("effective_batch_size")
    if effective is None and micro and accum:
        effective = micro * accum
    usable = pf.get("usable_gb")
    limit = pf.get("limit_gb")
    probe_peak = _selected_probe_peak(pf)
    gap = (actual_res - probe_peak
           if actual_res is not None and probe_peak is not None else None)
    margin = limit - actual_res if limit is not None and actual_res is not None else None

    return {
        "run": os.path.basename(os.path.dirname(path)),
        "mode": pf.get("mode", "?"),
        "micro": micro,
        "effective": effective,
        "limit": limit,
        "probe_peak": probe_peak,
        "actual_res": actual_res,
        "actual_alc": actual_alc,
        "gap": gap,
        "margin": margin,
        "bias_err": None,
        "verdict": _new_verdict(actual_res, usable, limit),
    }


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="src/output")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.output_dir)
    if not runs:
        print(f"No peak_memory.json found under {args.output_dir}")
        return 1

    rows = [evaluate(p, d) for p, d in runs]
    hdr = ("run", "mode", "micro", "eff", "limit", "probe", "actual",
           "gap", "margin", "verdict")
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(_fmt(r[k]) for k in (
            "run", "mode", "micro", "effective", "limit", "probe_peak",
            "actual_res", "gap", "margin", "verdict")))

    new = [r for r in rows if r["mode"] != "legacy"]
    print(f"\n{len(rows)} runs, {len(new)} with measured preflight; "
          "legacy runs predate real-Trainer probing.")
    bad = [r for r in rows
           if r["verdict"] in ("OOM/OVER", "OVER-USABLE", "OVER-LIMIT",
                               "BIAS-OFF")]
    if args.strict and bad:
        print(f"STRICT FAIL: {[r['run'] for r in bad]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
