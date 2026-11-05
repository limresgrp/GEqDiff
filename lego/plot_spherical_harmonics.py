#!/usr/bin/env python3
"""Generate didactic Plotly pages for LEGO spherical-harmonic shape features."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lego.lattice import _block_type_signature
from lego.lego_blocks import LEGO_LIBRARY
from lego.utils import build_surface_mesh, spherical_harmonic_basis


BLOCK_ORDER = ("1x1", "1x2", "L-shape", "T-shape")
DEGREE_SLICES = (slice(0, 1), slice(1, 4), slice(4, 9), slice(9, 16))
CAMERA = {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}, "projection": {"type": "orthographic"}}


def _sphere_grid(resolution: int):
    theta = np.linspace(0.0, np.pi, resolution)
    phi = np.linspace(0.0, 2.0 * np.pi, 2 * resolution, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    directions = np.stack(
        (
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ),
        axis=-1,
    )
    return theta_grid, phi_grid, directions


def _scene_style():
    axis = {"visible": False, "showbackground": False, "showgrid": False, "zeroline": False}
    return {
        "xaxis": axis,
        "yaxis": axis,
        "zaxis": axis,
        "aspectmode": "data",
        "camera": CAMERA,
        "bgcolor": "rgba(0,0,0,0)",
    }


def _style_figure(fig: go.Figure, title: str, height: int) -> None:
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        height=height,
        margin={"l": 12, "r": 12, "t": 92, "b": 18},
        paper_bgcolor="#f4f0e6",
        font={"family": "Avenir Next, Trebuchet MS, sans-serif", "color": "#18332f"},
        showlegend=False,
    )
    for key in fig.layout:
        if str(key).startswith("scene"):
            fig.update_layout({key: _scene_style()})


def _basis_figure(resolution: int) -> go.Figure:
    theta, phi, directions = _sphere_grid(resolution)
    basis = spherical_harmonic_basis(directions.reshape(-1, 3), lmax=3).reshape(*theta.shape, 16)
    labels = [f"Y({degree},{order:+d})" for degree in range(4) for order in range(-degree, degree + 1)]
    fig = make_subplots(
        rows=4,
        cols=4,
        specs=[[{"type": "surface"}] * 4 for _ in range(4)],
        subplot_titles=labels,
        horizontal_spacing=0.015,
        vertical_spacing=0.055,
    )
    for index, label in enumerate(labels):
        values = basis[..., index]
        amplitude = float(np.max(np.abs(values)))
        signed = values / max(amplitude, 1e-8)
        radius = np.abs(signed)
        row, col = divmod(index, 4)
        fig.add_trace(
            go.Surface(
                x=radius * np.sin(theta) * np.cos(phi),
                y=radius * np.sin(theta) * np.sin(phi),
                z=radius * np.cos(theta),
                surfacecolor=signed,
                colorscale=[[0.0, "#176b87"], [0.5, "#f7f3e8"], [1.0, "#d34b36"]],
                cmin=-1.0,
                cmax=1.0,
                showscale=index == 3,
                colorbar={"title": "sign", "len": 0.2, "y": 0.89},
                hovertemplate=f"{label}<br>normalized value=%{{surfacecolor:.3f}}<extra></extra>",
            ),
            row=row + 1,
            col=col + 1,
        )
    _style_figure(fig, "The 16 real spherical-harmonic basis functions through l = 3", 1180)
    return fig


def _shape_surface(coefficients: np.ndarray, resolution: int, showscale: bool = False) -> go.Surface:
    x, y, z, radii = build_surface_mesh(
        coefficients,
        resolution=resolution,
        base_radius=0.55,
        radial_scale=0.16,
        min_radius=0.12,
        max_radius=1.05,
    )
    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=radii,
        colorscale=[[0.0, "#cae5df"], [0.5, "#52a69a"], [1.0, "#ef8354"]],
        cmin=0.12,
        cmax=1.05,
        showscale=showscale,
        colorbar={"title": "radius", "len": 0.25},
        lighting={"ambient": 0.72, "diffuse": 0.75, "specular": 0.12, "roughness": 0.85},
        hovertemplate="radius=%{surfacecolor:.3f}<extra></extra>",
    )


def _combination_figure(resolution: int):
    titles = [f"{block}<br>degrees 0...{degree}" for block in BLOCK_ORDER for degree in range(4)]
    fig = make_subplots(
        rows=len(BLOCK_ORDER),
        cols=4,
        specs=[[{"type": "surface"}] * 4 for _ in BLOCK_ORDER],
        subplot_titles=titles,
        horizontal_spacing=0.018,
        vertical_spacing=0.06,
    )
    coefficient_rows = []
    for row, block in enumerate(BLOCK_ORDER, start=1):
        coefficients = _block_type_signature(block, np.eye(3, dtype=np.float32))
        coefficient_rows.append((block, coefficients))
        for degree in range(4):
            cumulative = coefficients.copy()
            cumulative[DEGREE_SLICES[degree].stop:] = 0.0
            fig.add_trace(_shape_surface(cumulative, resolution), row=row, col=degree + 1)
    _style_figure(fig, "Weighted sums: adding l = 0, 1, 2, and 3 information", 1120)
    return fig, coefficient_rows


def _cube_mesh(offsets: np.ndarray, color: str) -> go.Mesh3d:
    corners = np.asarray(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=np.float32,
    )
    triangles = np.asarray(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int32,
    )
    vertices = []
    faces = []
    for offset in offsets:
        base = len(vertices)
        vertices.extend(corners + offset)
        faces.extend(triangles + base)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        flatshading=True,
        opacity=0.96,
        lighting={"ambient": 0.7, "diffuse": 0.8, "specular": 0.08, "roughness": 0.9},
        hovertemplate="voxel center=(%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>",
    )


def _assignment_figure(resolution: int) -> go.Figure:
    titles = [label for block in BLOCK_ORDER for label in (f"{block}: SH surface", f"{block}: assigned voxels")]
    fig = make_subplots(
        rows=len(BLOCK_ORDER),
        cols=2,
        specs=[[{"type": "surface"}, {"type": "mesh3d"}] for _ in BLOCK_ORDER],
        subplot_titles=titles,
        horizontal_spacing=0.025,
        vertical_spacing=0.06,
    )
    for row, block in enumerate(BLOCK_ORDER, start=1):
        offsets = np.asarray(LEGO_LIBRARY[block]["offsets"], dtype=np.float32)
        coefficients = _block_type_signature(block, np.eye(3, dtype=np.float32))
        fig.add_trace(_shape_surface(coefficients, resolution), row=row, col=1)
        fig.add_trace(_cube_mesh(offsets, LEGO_LIBRARY[block]["color"]), row=row, col=2)
    _style_figure(fig, "From a smooth multipole signature to a discrete LEGO block", 1320)
    return fig


def _coefficient_table(rows) -> str:
    header = "".join(f"<th>c{i}</th>" for i in range(16))
    body = []
    for block, coefficients in rows:
        cells = "".join(f"<td>{value:.3f}</td>" for value in coefficients)
        body.append(f"<tr><th>{html.escape(block)}</th>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr><th>brick</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _write_page(path: Path, title: str, kicker: str, explanation: str, figure: go.Figure, extra: str, include_js: str):
    plot = pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=True if include_js == "inline" else "cdn",
        config={"responsive": True, "displaylogo": False},
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--paper:#f4f0e6;--ink:#18332f;--accent:#d65f3f;--line:#b8c9c3}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}}
header{{max-width:1120px;margin:0 auto;padding:54px 28px 20px}} .kicker{{color:var(--accent);font-weight:800;letter-spacing:.14em;text-transform:uppercase;font-size:.78rem}}
h1{{font-family:Georgia,serif;font-size:clamp(2rem,5vw,4.2rem);line-height:1;margin:.25em 0}} p{{max-width:850px;line-height:1.65;font-size:1.05rem}}
nav a{{color:var(--ink);font-weight:700;margin-right:20px}} main{{padding:0 18px 56px}} .plot{{max-width:1440px;margin:auto;border:1px solid var(--line);background:#fff;box-shadow:0 18px 45px #18332f18}}
.notes{{max-width:1120px;margin:28px auto;padding:0 10px}} .table-wrap{{overflow-x:auto;border:1px solid var(--line);background:#fff}} table{{border-collapse:collapse;font-size:.72rem}} th,td{{padding:7px;border-bottom:1px solid #dde5e1;text-align:right;white-space:nowrap}} th:first-child{{text-align:left;position:sticky;left:0;background:#fff}}
</style></head><body><header><div class="kicker">{html.escape(kicker)}</div><h1>{html.escape(title)}</h1><p>{explanation}</p>
<nav><a href="index.html">Overview</a><a href="01_basis_functions.html">Basis</a><a href="02_weighted_combinations.html">Combinations</a><a href="03_lego_assignments.html">LEGO mapping</a></nav></header>
<main><div class="plot">{plot}</div><div class="notes">{extra}</div></main></body></html>"""
    path.write_text(page, encoding="utf-8")


def _write_index(output_dir: Path):
    cards = (
        ("01", "Basis functions", "See every real Y(l,m) component through l=3 as a signed 3D lobe.", "01_basis_functions.html"),
        ("02", "Weighted combinations", "Add successive angular degrees and watch each LEGO signature emerge.", "02_weighted_combinations.html"),
        ("03", "LEGO assignment", "Compare each continuous SH surface with the discrete block it encodes.", "03_lego_assignments.html"),
    )
    card_html = "".join(
        f'<a class="card" href="{href}"><b>{number}</b><h2>{title}</h2><p>{text}</p></a>'
        for number, title, text, href in cards
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spherical harmonics for LEGO shapes</title><style>
:root{{--paper:#f4f0e6;--ink:#18332f;--accent:#d65f3f;--mint:#a8d5c8}} *{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}} main{{max-width:1160px;margin:auto;padding:9vh 28px}}
.kicker{{color:var(--accent);font-weight:800;letter-spacing:.14em;text-transform:uppercase}} h1{{font:clamp(2.8rem,8vw,6.8rem)/.93 Georgia,serif;max-width:900px;margin:.2em 0 .3em}}
.lead{{font-size:1.2rem;line-height:1.65;max-width:800px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px;margin-top:55px}}
.card{{display:block;min-height:260px;padding:28px;color:inherit;text-decoration:none;border:1px solid #9db8af;background:#fff;transition:transform .2s,box-shadow .2s}}
.card:hover{{transform:translateY(-5px);box-shadow:0 18px 38px #18332f20}} .card b{{color:var(--accent);font-size:2rem}} .card h2{{font:1.8rem Georgia,serif;margin:1.2em 0 .4em}} .card p{{line-height:1.55}}
</style></head><body><main><div class="kicker">GEqDiff visual primer</div><h1>How harmonics become bricks.</h1>
<p class="lead">The LEGO dataset stores shape as 16 real spherical-harmonic coefficients. These pages move from individual angular patterns, through weighted sums, to the four discrete blocks used by the generator.</p>
<div class="grid">{card_html}</div></main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tutorials/spherical_harmonics"))
    parser.add_argument("--resolution", type=int, default=28, help="Latitude samples per surface (default: 28).")
    parser.add_argument("--plotlyjs", choices=("cdn", "inline"), default="cdn")
    args = parser.parse_args(argv)
    if args.resolution < 12:
        parser.error("--resolution must be at least 12.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    basis = _basis_figure(args.resolution)
    combinations, rows = _combination_figure(args.resolution)
    assignments = _assignment_figure(args.resolution)
    _write_index(args.output_dir)
    _write_page(
        args.output_dir / "01_basis_functions.html",
        "The angular alphabet",
        "Page 1 / Basis",
        "Each surface is one real e3nn spherical harmonic. Radius shows absolute amplitude; blue and red distinguish opposite signs. A coefficient scales one of these patterns before it enters a sum.",
        basis,
        "<p>The coefficient order is grouped by degree: 1 scalar for l=0, 3 components for l=1, 5 for l=2, and 7 for l=3. The basis uses integral normalization, matching the dataset.</p>",
        args.plotlyjs,
    )
    _write_page(
        args.output_dir / "02_weighted_combinations.html",
        "Shape by accumulation",
        "Page 2 / Combination",
        "Every row starts with the spherical l=0 term, then cumulatively introduces directional l=1, quadrupolar l=2, and finer l=3 detail using the actual LEGO coefficients.",
        combinations,
        "<h2>Canonical coefficient vectors</h2>" + _coefficient_table(rows),
        args.plotlyjs,
    )
    _write_page(
        args.output_dir / "03_lego_assignments.html",
        "Continuous signal, discrete brick",
        "Page 3 / Assignment",
        "The left column reconstructs the smooth radial field used by the visualizer. The right column shows the canonical voxel block whose occupied offsets generated that 16D signature.",
        assignments,
        "<p>Rotating a brick rotates its non-scalar coefficients equivariantly. The discrete type remains the same, while the reconstructed surface turns with the physical block.</p>",
        args.plotlyjs,
    )
    print(f"Wrote spherical-harmonic tutorial to {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
