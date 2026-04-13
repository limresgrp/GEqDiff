from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "heun100_noguide",
        "args": ["--steps", "100", "--sampler", "heun", "--no-clash-guidance"],
    },
    {
        "name": "heun100_clash",
        "args": [
            "--steps",
            "100",
            "--sampler",
            "heun",
            "--clash-guidance",
            "--clash-guidance-strength",
            "0.05",
            "--clash-guidance-weight-schedule",
            "late_quadratic",
            "--clash-guidance-auto-scale",
            "--clash-guidance-auto-scale-min",
            "0.2",
            "--clash-guidance-auto-scale-max",
            "5.0",
        ],
    },
    {
        "name": "heun100_clash_refine",
        "args": [
            "--steps",
            "100",
            "--sampler",
            "heun",
            "--clash-guidance",
            "--clash-guidance-strength",
            "0.05",
            "--clash-guidance-weight-schedule",
            "late_quadratic",
            "--clash-guidance-auto-scale",
            "--clash-guidance-auto-scale-min",
            "0.2",
            "--clash-guidance-auto-scale-max",
            "5.0",
            "--late-refine-from-step",
            "10",
            "--late-refine-factor",
            "4",
        ],
    },
    {
        "name": "heun100_clash_cohesion",
        "args": [
            "--steps",
            "100",
            "--sampler",
            "heun",
            "--clash-guidance",
            "--clash-guidance-strength",
            "0.05",
            "--clash-guidance-weight-schedule",
            "late_quadratic",
            "--clash-guidance-auto-scale",
            "--clash-guidance-auto-scale-min",
            "0.2",
            "--clash-guidance-auto-scale-max",
            "5.0",
            "--cohesion-guidance-strength",
            "0.50",
            "--cohesion-guidance-target-contacts",
            "1.5",
        ],
    },
    {
        "name": "heun100_clash_cohesion_refine",
        "args": [
            "--steps",
            "100",
            "--sampler",
            "heun",
            "--clash-guidance",
            "--clash-guidance-strength",
            "0.05",
            "--clash-guidance-weight-schedule",
            "late_quadratic",
            "--clash-guidance-auto-scale",
            "--clash-guidance-auto-scale-min",
            "0.2",
            "--clash-guidance-auto-scale-max",
            "5.0",
            "--cohesion-guidance-strength",
            "0.50",
            "--cohesion-guidance-target-contacts",
            "1.5",
            "--late-refine-from-step",
            "10",
            "--late-refine-factor",
            "4",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small LEGO sampling sweep across checkpoints and sampler settings.")
    parser.add_argument("--dataset", type=Path, required=True, help="Diffusion dataset NPZ used for conditioning/sampling.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where sampled NPZs, metrics, and reports are written.")
    parser.add_argument("--models", type=Path, nargs="+", required=True, help="Checkpoint paths to compare.")
    parser.add_argument("--indices", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7], help="Fixed diffusion example indices for the comparative sweep.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for sampling.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed forwarded to sampling.")
    parser.add_argument("--source-canonical", type=Path, default=None, help="Optional canonical source dataset for mesh/target enrichment.")
    return parser.parse_args()


def _safe_name(path: Path) -> str:
    return path.parent.name + "__" + path.stem


def _run_sampling(
    *,
    root: Path,
    model: Path,
    dataset: Path,
    out_npz: Path,
    config_args: List[str],
    indices: List[int],
    device: str,
    seed: int,
    source_canonical: Path | None,
) -> None:
    cmd = [
        sys.executable,
        str(root / "geqdiff/scripts/sample_lego.py"),
        "--model",
        str(model),
        "--input",
        str(dataset),
        "--output",
        str(out_npz),
        "--device",
        device,
        "--seed",
        str(seed),
        "--save-metrics",
        "--indices",
        *[str(idx) for idx in indices],
        *config_args,
    ]
    if source_canonical is not None:
        cmd.extend(["--source-canonical", str(source_canonical)])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "deps/GEqTrain") + os.pathsep + str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(root), env=env)


def _load_summary(metrics_path: Path) -> Dict[str, Any]:
    with metrics_path.open() as handle:
        report = json.load(handle)
    return report["summary"]


def _finite_or_zero(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _ranking_key(row: Dict[str, Any]) -> tuple:
    return (
        int(row["valid_like_geometries"]),
        _finite_or_zero(row["mean_validity_score"]),
        -_finite_or_zero(row["mean_diffused_shift"]),
        -_finite_or_zero(row["mean_energy_delta"]),
    )


def _write_markdown_report(report_path: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# LEGO Sampling Sweep",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Device: `{args.device}`",
        f"- Indices: `{' '.join(str(i) for i in args.indices)}`",
        "",
        "| Rank | Model | Config | Valid-like | Mean validity | Mean overlap | Mean components | Mean shift | Mean energy delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(rows, key=_ranking_key, reverse=True)
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + f"{rank} | `{row['model_name']}` | `{row['config_name']}` | "
            + f"{row['valid_like_geometries']}/{row['num_samples']} | "
            + f"{row['mean_validity_score']:.3f} | "
            + f"{row['mean_overlap']:.3f} | "
            + f"{row['mean_components']:.3f} | "
            + f"{row['mean_diffused_shift']:.3f} | "
            + f"{row['mean_energy_delta']:.3f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for model in args.models:
        model_name = _safe_name(model)
        model_dir = args.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for config in DEFAULT_CONFIGS:
            config_name = str(config["name"])
            out_npz = model_dir / f"{config_name}.npz"
            _run_sampling(
                root=root,
                model=model,
                dataset=args.dataset,
                out_npz=out_npz,
                config_args=list(config["args"]),
                indices=list(args.indices),
                device=args.device,
                seed=int(args.seed),
                source_canonical=args.source_canonical,
            )
            metrics_path = out_npz.with_name(f"{out_npz.stem}_metrics.json")
            summary = _load_summary(metrics_path)
            with metrics_path.open() as handle:
                report = json.load(handle)
            records = report["records"]
            mean_overlap = sum(r["score_card"]["sampled"]["metrics"]["total_overlap_volume"] for r in records) / max(len(records), 1)
            mean_components = sum(r["score_card"]["sampled"]["metrics"]["num_components"] for r in records) / max(len(records), 1)
            rows.append(
                {
                    "model_name": model_name,
                    "model_path": str(model),
                    "config_name": config_name,
                    "output_npz": str(out_npz),
                    "metrics_json": str(metrics_path),
                    "num_samples": int(summary["num_samples"]),
                    "valid_like_geometries": int(summary["valid_like_geometries"]),
                    "mean_validity_score": _finite_or_zero(summary["mean_validity_score"]),
                    "mean_diffused_shift": _finite_or_zero(summary["mean_diffused_shift"]),
                    "mean_energy_delta": _finite_or_zero(summary["mean_energy_delta"]),
                    "mean_overlap": float(mean_overlap),
                    "mean_components": float(mean_components),
                }
            )

    json_path = args.output_dir / "sampling_sweep_report.json"
    md_path = args.output_dir / "sampling_sweep_report.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    _write_markdown_report(md_path, rows, args)
    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")


if __name__ == "__main__":
    main()
