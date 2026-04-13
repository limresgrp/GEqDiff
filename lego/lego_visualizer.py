from __future__ import annotations

import argparse
import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder

try:
    from lego.lego_blocks import LEGO_LIBRARY, rotated_offsets
    from lego.score_utils import evaluate_sample_scores
    from lego.utils import block_palette, build_surface_mesh, default_dataset_path, load_samples
except ModuleNotFoundError:
    from lego_blocks import LEGO_LIBRARY, rotated_offsets
    from score_utils import evaluate_sample_scores
    from utils import block_palette, build_surface_mesh, default_dataset_path, load_samples


FACE_DIRECTIONS = (
    np.asarray((1, 0, 0), dtype=np.int32),
    np.asarray((-1, 0, 0), dtype=np.int32),
    np.asarray((0, 1, 0), dtype=np.int32),
    np.asarray((0, -1, 0), dtype=np.int32),
    np.asarray((0, 0, 1), dtype=np.int32),
    np.asarray((0, 0, -1), dtype=np.int32),
)
NEGATIVE_COLOR = np.asarray((205, 72, 72), dtype=np.float32)
NEUTRAL_COLOR = np.asarray((214, 214, 214), dtype=np.float32)
POSITIVE_COLOR = np.asarray((74, 118, 212), dtype=np.float32)
DEFAULT_DIPOLE_LENGTH = 0.9
MIN_DIPOLE_LENGTH = 0.2
MAX_DIPOLE_LENGTH = 2.2
DEFAULT_BLOCK_OPACITY = 1.0
MIN_BLOCK_OPACITY = 0.15
MAX_BLOCK_OPACITY = 1.0
SAMPLED_STRUCTURE_FIELDS = (
    "brick_anchors",
    "brick_types",
    "brick_rotations",
    "brick_features",
    "brick_dipoles",
    "brick_dipole_strengths",
    "brick_dipole_directions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize canonical LEGO datasets with brick meshes, SH node surfaces, dipoles, "
            "and the smooth SH target."
        )
    )
    parser.add_argument("--path", type=Path, default=default_dataset_path(), help="Canonical LEGO NPZ to visualize.")
    parser.add_argument("--index", type=int, default=0, help="Initial sample index.")
    parser.add_argument(
        "--structure-view",
        type=str,
        default="sampled",
        choices=["sampled", "original"],
        help="Initial structure to display when both sampled and original fields are present.",
    )
    parser.add_argument(
        "--display-view",
        type=str,
        default="bricks",
        choices=["bricks", "surfaces"],
        help="Initial geometry view: full brick surfaces or per-node SH surfaces.",
    )
    parser.add_argument(
        "--target-view",
        type=str,
        default="hidden",
        choices=["hidden", "wireframe", "filled", "surface", "voxels"],
        help="Initial target view. `surface` is kept as a compatibility alias for `filled`.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="orthographic",
        choices=["orthographic", "perspective"],
        help="Scene projection. Orthographic avoids perspective ordering artifacts.",
    )
    parser.add_argument(
        "--show-dipoles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show dipole axes on top of the current structure.",
    )
    parser.add_argument(
        "--show-ports",
        action="store_true",
        help="Deprecated compatibility alias for `--show-dipoles`.",
    )
    parser.add_argument(
        "--show-target-voxels",
        action="store_true",
        help="Compatibility flag that sets the initial target view to `voxels`.",
    )
    parser.add_argument("--output-html", type=Path, default=None, help="Optional HTML output path.")
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=1,
        help="If intermediate states are present, keep every N-th trajectory frame when building the HTML.",
    )
    parser.add_argument("--no-show", action="store_true", help="Write the HTML but do not try to open it.")
    return parser.parse_args()


def _as_int_rotation(rotation: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(rotation, dtype=np.float32)).astype(np.int32)


def _brick_offsets(brick_type: str, rotation: np.ndarray) -> np.ndarray:
    offsets = np.asarray(LEGO_LIBRARY[str(brick_type)]["offsets"], dtype=np.int32)
    return rotated_offsets(offsets, _as_int_rotation(rotation)).astype(np.int32)


def _brick_center(anchor: np.ndarray, brick_type: str, rotation: np.ndarray) -> np.ndarray:
    offsets = _brick_offsets(brick_type, rotation).astype(np.float32)
    return np.asarray(anchor, dtype=np.float32) + offsets.mean(axis=0)


def _brick_voxels(anchor: np.ndarray, brick_type: str, rotation: np.ndarray) -> np.ndarray:
    return _brick_offsets(brick_type, rotation).astype(np.float32) + np.asarray(anchor, dtype=np.float32)


def _hex_color(rgb: np.ndarray) -> str:
    rgb = np.clip(np.rint(np.asarray(rgb, dtype=np.float32)), 0, 255).astype(np.int32)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _mix_color(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * np.asarray(a, dtype=np.float32) + t * np.asarray(b, dtype=np.float32)


def _dipole_face_color(face_center: np.ndarray, brick_center: np.ndarray, dipole: np.ndarray) -> str:
    dipole = np.asarray(dipole, dtype=np.float32)
    strength = float(np.linalg.norm(dipole))
    if strength <= 1e-8:
        return _hex_color(NEUTRAL_COLOR)

    direction = dipole / strength
    rel = np.asarray(face_center, dtype=np.float32) - np.asarray(brick_center, dtype=np.float32)
    rel_norm = float(np.linalg.norm(rel))
    if rel_norm <= 1e-8:
        signed_value = 0.0
    else:
        signed_value = float(np.dot(direction, rel / rel_norm))
    signed_value = float(np.clip(signed_value * min(strength, 1.0), -1.0, 1.0))

    if signed_value >= 0.0:
        color = _mix_color(NEUTRAL_COLOR, POSITIVE_COLOR, signed_value)
    else:
        color = _mix_color(NEUTRAL_COLOR, NEGATIVE_COLOR, -signed_value)
    return _hex_color(color)


def _orthogonal_face_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx, dy, dz = np.asarray(direction, dtype=np.float32).tolist()
    if abs(dx) > 0:
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32), np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    if abs(dy) > 0:
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float32), np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return np.asarray((1.0, 0.0, 0.0), dtype=np.float32), np.asarray((0.0, 1.0, 0.0), dtype=np.float32)


def _face_vertices(voxel_center: np.ndarray, direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32)
    u_axis, v_axis = _orthogonal_face_basis(direction)
    face_center = np.asarray(voxel_center, dtype=np.float32) + 0.5 * direction
    return np.asarray(
        [
            face_center - 0.5 * u_axis - 0.5 * v_axis,
            face_center + 0.5 * u_axis - 0.5 * v_axis,
            face_center + 0.5 * u_axis + 0.5 * v_axis,
            face_center - 0.5 * u_axis + 0.5 * v_axis,
        ],
        dtype=np.float32,
    )


def _paired_legend_group(sample: Dict, brick_index: int) -> str:
    mask = np.asarray(sample.get("sampled_brick_mask", np.zeros((0,), dtype=bool)), dtype=bool).reshape(-1)
    if mask.size == 0:
        return f"brick_pair_{brick_index:03d}"
    prefix = "diffused" if bool(mask[brick_index]) else "fixed"
    return f"{prefix}_brick_pair_{brick_index:03d}"


def _brick_name(sample: Dict, brick_index: int, brick_type: str, dipole_strength: float) -> str:
    mask = np.asarray(sample.get("sampled_brick_mask", np.zeros((0,), dtype=bool)), dtype=bool).reshape(-1)
    prefix = "Diffused" if mask.size > 0 and bool(mask[brick_index]) else "Fixed" if mask.size > 0 else "Brick"
    dipole_label = "polar" if dipole_strength > 1e-6 else "neutral"
    return f"{prefix} {brick_index + 1:02d}: {brick_type} ({dipole_label})"


def _mesh_trace_from_brick(
    sample: Dict,
    anchors: np.ndarray,
    types: np.ndarray,
    rotations: np.ndarray,
    dipoles: np.ndarray,
    brick_index: int,
) -> go.BaseTraceType:
    anchor = np.asarray(anchors[brick_index], dtype=np.float32)
    brick_type = str(types[brick_index])
    rotation = np.asarray(rotations[brick_index], dtype=np.float32)
    dipole = np.asarray(dipoles[brick_index], dtype=np.float32)
    brick_center = _brick_center(anchor, brick_type, rotation)
    voxels = _brick_offsets(brick_type, rotation)
    occupied = {tuple(voxel) for voxel in voxels.tolist()}

    x: List[float] = []
    y: List[float] = []
    z: List[float] = []
    i: List[int] = []
    j: List[int] = []
    k: List[int] = []
    facecolor: List[str] = []

    for voxel in voxels.astype(np.float32):
        world_voxel = voxel + anchor
        for direction in FACE_DIRECTIONS:
            neighbor = tuple((voxel.astype(np.int32) + direction).tolist())
            if neighbor in occupied:
                continue
            corners = _face_vertices(world_voxel, direction)
            start = len(x)
            x.extend(corners[:, 0].tolist())
            y.extend(corners[:, 1].tolist())
            z.extend(corners[:, 2].tolist())
            i.extend([start + 0, start + 0])
            j.extend([start + 1, start + 2])
            k.extend([start + 2, start + 3])
            color = _dipole_face_color(world_voxel + 0.5 * direction.astype(np.float32), brick_center, dipole)
            facecolor.extend([color, color])

    group_key = _paired_legend_group(sample, brick_index)
    strength = float(np.linalg.norm(dipole))
    name = _brick_name(sample, brick_index, brick_type, strength)
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        facecolor=facecolor,
        name=name,
        hovertemplate=(
            f"{name}<br>"
            f"Center: ({brick_center[0]:.2f}, {brick_center[1]:.2f}, {brick_center[2]:.2f})<br>"
            f"Dipole strength: {strength:.2f}<extra></extra>"
        ),
        legendgroup=group_key,
        showlegend=True,
        flatshading=True,
        opacity=1.0,
        uid=f"brick-mesh-{brick_index}",
        lighting={"ambient": 0.78, "diffuse": 0.7, "specular": 0.1, "roughness": 0.9},
    )


def _surface_trace_from_brick(
    sample: Dict,
    anchors: np.ndarray,
    types: np.ndarray,
    rotations: np.ndarray,
    features: np.ndarray,
    dipoles: np.ndarray,
    brick_index: int,
) -> go.BaseTraceType:
    anchor = np.asarray(anchors[brick_index], dtype=np.float32)
    brick_type = str(types[brick_index])
    rotation = np.asarray(rotations[brick_index], dtype=np.float32)
    coefficients = np.asarray(features[brick_index], dtype=np.float32)
    dipole = np.asarray(dipoles[brick_index], dtype=np.float32)
    brick_center = _brick_center(anchor, brick_type, rotation)

    mesh_x, mesh_y, mesh_z, _ = build_surface_mesh(
        coefficients,
        resolution=18,
        base_radius=0.55,
        radial_scale=0.16,
        min_radius=0.12,
        max_radius=1.05,
    )
    mesh_x = mesh_x + brick_center[0]
    mesh_y = mesh_y + brick_center[1]
    mesh_z = mesh_z + brick_center[2]

    palette = block_palette()
    base_color = palette.get(brick_type, "#808080")
    surfacecolor = np.zeros_like(mesh_x, dtype=np.float32)
    group_key = _paired_legend_group(sample, brick_index)
    strength = float(np.linalg.norm(dipole))
    name = _brick_name(sample, brick_index, brick_type, strength)
    return go.Surface(
        x=mesh_x,
        y=mesh_y,
        z=mesh_z,
        surfacecolor=surfacecolor,
        colorscale=[[0.0, base_color], [1.0, base_color]],
        cmin=0.0,
        cmax=1.0,
        showscale=False,
        opacity=0.92,
        uid=f"brick-surface-{brick_index}",
        name=name,
        hovertemplate=f"{name}<extra></extra>",
        legendgroup=group_key,
        showlegend=True,
        contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
    )


def _dipole_specs(sample: Dict, anchors: np.ndarray, types: np.ndarray, rotations: np.ndarray, dipoles: np.ndarray) -> List[Dict]:
    mask = np.asarray(sample.get("sampled_brick_mask", np.zeros((len(anchors),), dtype=bool)), dtype=bool).reshape(-1)
    if mask.size == 0:
        mask = np.zeros((len(anchors),), dtype=bool)

    specs: List[Dict] = []
    for group_name, group_mask, color in (
        ("Fixed dipoles", ~mask, "#2f2f2f"),
        ("Diffused dipoles", mask, "#111111"),
    ):
        centers: List[List[float]] = []
        directions: List[List[float]] = []
        strengths: List[float] = []
        for brick_index in range(len(anchors)):
            if not bool(group_mask[brick_index]):
                continue
            dipole = np.asarray(dipoles[brick_index], dtype=np.float32)
            strength = float(np.linalg.norm(dipole))
            if strength <= 1e-8:
                continue
            brick_center = _brick_center(anchors[brick_index], str(types[brick_index]), rotations[brick_index])
            direction = dipole / strength
            centers.append(brick_center.astype(float).tolist())
            directions.append(direction.astype(float).tolist())
            strengths.append(strength)
        if len(centers) == 0:
            continue
        specs.append(
            {
                "name": group_name,
                "color": color,
                "centers": centers,
                "directions": directions,
                "strengths": strengths,
            }
        )
    return specs


def _dipole_traces_from_specs(specs: Sequence[Dict], length_scale: float) -> List[go.BaseTraceType]:
    traces: List[go.BaseTraceType] = []
    for spec in specs:
        centers = np.asarray(spec.get("centers", []), dtype=np.float32)
        directions = np.asarray(spec.get("directions", []), dtype=np.float32)
        strengths = np.asarray(spec.get("strengths", []), dtype=np.float32).reshape(-1)
        if centers.size == 0 or directions.size == 0 or strengths.size == 0:
            continue

        line_x: List[float | None] = []
        line_y: List[float | None] = []
        line_z: List[float | None] = []
        cone_x: List[float] = []
        cone_y: List[float] = []
        cone_z: List[float] = []
        cone_u: List[float] = []
        cone_v: List[float] = []
        cone_w: List[float] = []
        cone_ref = max(0.12, 0.55 * max(0.12, 0.28 * float(length_scale)))
        for center, direction, strength in zip(centers, directions, strengths):
            arrow_length = max(0.0, float(length_scale) * float(strength))
            if arrow_length <= 1e-8:
                continue
            cone_length = min(max(0.12, 0.28 * float(length_scale)), 0.65 * arrow_length)
            tip = center + 0.5 * arrow_length * direction
            start = center - 0.5 * arrow_length * direction
            shaft_end = tip - cone_length * direction
            line_x.extend([float(start[0]), float(shaft_end[0]), None])
            line_y.extend([float(start[1]), float(shaft_end[1]), None])
            line_z.extend([float(start[2]), float(shaft_end[2]), None])
            cone_x.append(float(tip[0]))
            cone_y.append(float(tip[1]))
            cone_z.append(float(tip[2]))
            cone_u.append(float(cone_length * direction[0]))
            cone_v.append(float(cone_length * direction[1]))
            cone_w.append(float(cone_length * direction[2]))

        if len(cone_x) == 0:
            continue

        color = str(spec.get("color", "#222222"))
        name = str(spec.get("name", "Dipoles"))
        traces.append(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"width": 7, "color": color},
                name=name,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Cone(
                x=cone_x,
                y=cone_y,
                z=cone_z,
                u=cone_u,
                v=cone_v,
                w=cone_w,
                anchor="tip",
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
                colorscale=[[0.0, color], [1.0, color]],
                cmin=0.0,
                cmax=1.0,
                sizemode="absolute",
                sizeref=cone_ref,
                name=name,
            )
        )
    return traces


def _wireframe_trace(mesh_x: np.ndarray, mesh_y: np.ndarray, mesh_z: np.ndarray, color: str) -> go.BaseTraceType:
    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []

    for row in range(mesh_x.shape[0]):
        xs.extend(mesh_x[row].astype(float).tolist() + [None])
        ys.extend(mesh_y[row].astype(float).tolist() + [None])
        zs.extend(mesh_z[row].astype(float).tolist() + [None])
    for col in range(mesh_x.shape[1]):
        xs.extend(mesh_x[:, col].astype(float).tolist() + [None])
        ys.extend(mesh_y[:, col].astype(float).tolist() + [None])
        zs.extend(mesh_z[:, col].astype(float).tolist() + [None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"width": 3, "color": color},
        name="Smooth SH target",
        showlegend=True,
        hoverinfo="skip",
    )


def _target_trace_sets(sample: Dict) -> Dict[str, List[go.BaseTraceType]]:
    target: Dict[str, List[go.BaseTraceType]] = {
        "hidden": [],
        "wireframe": [],
        "filled": [],
        "voxels": [],
    }
    if "mesh_x" in sample and "mesh_y" in sample and "mesh_z" in sample:
        mesh_x = np.asarray(sample["mesh_x"], dtype=np.float32)
        mesh_y = np.asarray(sample["mesh_y"], dtype=np.float32)
        mesh_z = np.asarray(sample["mesh_z"], dtype=np.float32)
        filled = go.Surface(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            surfacecolor=np.zeros_like(mesh_x, dtype=np.float32),
            colorscale=[[0.0, "#6d7786"], [1.0, "#6d7786"]],
            cmin=0.0,
            cmax=1.0,
            showscale=False,
            opacity=0.24,
            name="Smooth SH target",
            showlegend=True,
            hovertemplate="Smooth SH target<extra></extra>",
        )
        wire = _wireframe_trace(mesh_x, mesh_y, mesh_z, color="#28313b")
        target["wireframe"] = [wire]
        target["filled"] = [filled, wire]

    if "target_voxels" in sample:
        voxels = np.asarray(sample["target_voxels"], dtype=np.float32)
        if voxels.size > 0:
            target["voxels"] = [
                go.Scatter3d(
                    x=voxels[:, 0],
                    y=voxels[:, 1],
                    z=voxels[:, 2],
                    mode="markers",
                    marker={"size": 4.5, "color": "#4c5665", "opacity": 0.38, "symbol": "square"},
                    name="Target voxels",
                    showlegend=True,
                    hovertemplate="Target voxel (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>",
                )
            ]
    return target


def _sample_with_sampled_structure(sample: Dict, sampled_structure: Dict) -> Dict:
    merged = dict(sample)
    for key in SAMPLED_STRUCTURE_FIELDS:
        if key in sampled_structure:
            merged[key] = np.asarray(sampled_structure[key]).copy()
    for key in ("stage_index", "stage_label", "scheduler_step", "tau"):
        if key in sampled_structure:
            merged[key] = np.asarray(sampled_structure[key]).copy()
    return merged


def _structure_traces(sample: Dict, prefix: str) -> Dict[str, List[go.BaseTraceType]]:
    anchors = np.asarray(sample[f"{prefix}brick_anchors"], dtype=np.float32)
    rotations = np.asarray(sample[f"{prefix}brick_rotations"], dtype=np.float32)
    types = np.asarray(sample[f"{prefix}brick_types"])
    features = np.asarray(sample[f"{prefix}brick_features"], dtype=np.float32)
    dipoles = np.asarray(sample.get(f"{prefix}brick_dipoles", np.zeros((len(anchors), 3), dtype=np.float32)), dtype=np.float32)

    bricks = [
        _mesh_trace_from_brick(sample, anchors=anchors, types=types, rotations=rotations, dipoles=dipoles, brick_index=brick_index)
        for brick_index in range(len(anchors))
    ]
    surfaces = [
        _surface_trace_from_brick(
            sample,
            anchors=anchors,
            types=types,
            rotations=rotations,
            features=features,
            dipoles=dipoles,
            brick_index=brick_index,
        )
        for brick_index in range(len(anchors))
    ]
    return {
        "bricks": bricks,
        "surfaces": surfaces,
        "dipole_specs": _dipole_specs(sample, anchors=anchors, types=types, rotations=rotations, dipoles=dipoles),
    }


def _structure_points(sample: Dict, prefix: str) -> np.ndarray:
    anchors_key = f"{prefix}brick_anchors"
    if anchors_key not in sample:
        return np.zeros((0, 3), dtype=np.float32)
    anchors = np.asarray(sample[anchors_key], dtype=np.float32)
    types = np.asarray(sample[f"{prefix}brick_types"])
    rotations = np.asarray(sample[f"{prefix}brick_rotations"], dtype=np.float32)
    points: List[np.ndarray] = []
    for anchor, brick_type, rotation in zip(anchors, types, rotations):
        voxels = _brick_voxels(anchor, str(brick_type), rotation)
        points.append(voxels)
        points.append(_brick_center(anchor, str(brick_type), rotation)[None, :])
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32)


def _target_points(sample: Dict) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for key in ("target_voxels",):
        if key in sample:
            values = np.asarray(sample[key], dtype=np.float32)
            if values.size > 0:
                chunks.append(values.reshape(-1, 3))
    if "mesh_x" in sample:
        chunks.append(
            np.stack(
                [
                    np.asarray(sample["mesh_x"], dtype=np.float32).reshape(-1),
                    np.asarray(sample["mesh_y"], dtype=np.float32).reshape(-1),
                    np.asarray(sample["mesh_z"], dtype=np.float32).reshape(-1),
                ],
                axis=-1,
            )
        )
    if len(chunks) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _scene_spec(sample: Dict, projection: str) -> Dict:
    trajectory_points: List[np.ndarray] = []
    for intermediate in sample.get("intermediate_states", []):
        stage_sample = _sample_with_sampled_structure(sample, intermediate)
        trajectory_points.append(_structure_points(stage_sample, ""))
    points = np.concatenate(
        [
            _structure_points(sample, ""),
            _structure_points(sample, "original_"),
            *trajectory_points,
            _target_points(sample),
        ],
        axis=0,
    )
    if points.size == 0:
        points = np.zeros((1, 3), dtype=np.float32)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = np.maximum(maxs - mins, 1.0)
    margin = 0.18 * float(extents.max()) + 0.75
    ranges = [
        [float(mins[0] - margin), float(maxs[0] + margin)],
        [float(mins[1] - margin), float(maxs[1] + margin)],
        [float(mins[2] - margin), float(maxs[2] + margin)],
    ]
    return {
        "xaxis": {"visible": False, "range": ranges[0], "showbackground": False},
        "yaxis": {"visible": False, "range": ranges[1], "showbackground": False},
        "zaxis": {"visible": False, "range": ranges[2], "showbackground": False},
        "aspectmode": "data",
        "uirevision": f"scene-{projection}",
        "camera": {
            "projection": {"type": projection},
            "eye": {"x": 1.6, "y": 1.45, "z": 1.25},
        },
    }


def _sample_meta(sample: Dict) -> Dict[str, int | bool | float | str]:
    anchors = np.asarray(sample["brick_anchors"], dtype=np.float32)
    dipoles = np.asarray(sample.get("brick_dipoles", np.zeros((len(anchors), 3), dtype=np.float32)), dtype=np.float32)
    mask = np.asarray(sample.get("sampled_brick_mask", np.zeros((len(anchors),), dtype=bool)), dtype=bool).reshape(-1)
    meta: Dict[str, int | bool | float | str] = {
        "num_bricks": int(len(anchors)),
        "num_polar": int((np.linalg.norm(dipoles, axis=-1) > 1e-6).sum()),
        "num_diffused": int(mask.sum()) if mask.size > 0 else 0,
        "has_original": bool("original_brick_anchors" in sample),
    }
    if "source_frame_id" in sample:
        meta["source_frame_id"] = int(np.asarray(sample["source_frame_id"]).reshape(-1)[0])
    if "split_id" in sample:
        meta["split_id"] = int(np.asarray(sample["split_id"]).reshape(-1)[0])
    if "occupancy_mode" in sample:
        meta["occupancy_mode"] = str(np.asarray(sample["occupancy_mode"]).reshape(-1)[0])
    if "shell_thickness" in sample:
        meta["shell_thickness"] = float(np.asarray(sample["shell_thickness"]).reshape(-1)[0])
    if "shell_sparsity" in sample:
        meta["shell_sparsity"] = float(np.asarray(sample["shell_sparsity"]).reshape(-1)[0])
    if "sampling_sampler" in sample:
        meta["sampling_sampler"] = str(np.asarray(sample["sampling_sampler"]).reshape(-1)[0])
    if "sampling_steps" in sample:
        meta["sampling_steps"] = int(np.asarray(sample["sampling_steps"]).reshape(-1)[0])
    if "sampling_late_refine_from_step" in sample:
        meta["sampling_late_refine_from_step"] = int(np.asarray(sample["sampling_late_refine_from_step"]).reshape(-1)[0])
    if "sampling_late_refine_factor" in sample:
        meta["sampling_late_refine_factor"] = int(np.asarray(sample["sampling_late_refine_factor"]).reshape(-1)[0])
    if "sampling_linger_step" in sample:
        meta["sampling_linger_step"] = int(np.asarray(sample["sampling_linger_step"]).reshape(-1)[0])
    if "sampling_linger_count" in sample:
        meta["sampling_linger_count"] = int(np.asarray(sample["sampling_linger_count"]).reshape(-1)[0])
    if "sampling_clash_guidance" in sample:
        meta["sampling_clash_guidance"] = bool(np.asarray(sample["sampling_clash_guidance"]).reshape(-1)[0])
    if "sampling_clash_guidance_strength" in sample:
        meta["sampling_clash_guidance_strength"] = float(np.asarray(sample["sampling_clash_guidance_strength"]).reshape(-1)[0])
    if "sampling_clash_guidance_max_norm" in sample:
        meta["sampling_clash_guidance_max_norm"] = float(np.asarray(sample["sampling_clash_guidance_max_norm"]).reshape(-1)[0])
    if "sampling_clash_guidance_weight_schedule" in sample:
        meta["sampling_clash_guidance_weight_schedule"] = str(np.asarray(sample["sampling_clash_guidance_weight_schedule"]).reshape(-1)[0])
    if "sampling_clash_guidance_auto_scale" in sample:
        meta["sampling_clash_guidance_auto_scale"] = bool(np.asarray(sample["sampling_clash_guidance_auto_scale"]).reshape(-1)[0])
    if "sampling_cohesion_guidance_strength" in sample:
        meta["sampling_cohesion_guidance_strength"] = float(np.asarray(sample["sampling_cohesion_guidance_strength"]).reshape(-1)[0])
    if "sampling_cohesion_guidance_target_contacts" in sample:
        meta["sampling_cohesion_guidance_target_contacts"] = float(np.asarray(sample["sampling_cohesion_guidance_target_contacts"]).reshape(-1)[0])
    if "intermediate_states" in sample:
        meta["num_intermediate_states"] = int(len(sample["intermediate_states"]))
    if "trajectory_stride" in sample:
        meta["trajectory_stride"] = int(np.asarray(sample["trajectory_stride"]).reshape(-1)[0])
    return meta


def _apply_trajectory_stride(sample: Dict, stride: int) -> Dict:
    stride = max(int(stride), 1)
    if stride <= 1 or "intermediate_states" not in sample:
        return sample

    raw_states = list(sample.get("intermediate_states", []))
    if len(raw_states) <= 2:
        return sample

    selected = raw_states[::stride]
    if selected[-1] is not raw_states[-1]:
        selected.append(raw_states[-1])

    updated = dict(sample)
    updated["intermediate_states"] = selected
    updated["trajectory_stride"] = np.asarray(stride, dtype=np.int64)
    updated["original_num_intermediate_states"] = np.asarray(len(raw_states), dtype=np.int64)
    return updated


def _trajectory_state_payloads(sample: Dict) -> List[Dict]:
    raw_stages = list(sample.get("intermediate_states", []))
    if len(raw_stages) == 0:
        raw_stages = [sample]

    payloads: List[Dict] = []
    for stage_index, raw_stage in enumerate(raw_stages):
        stage_sample = _sample_with_sampled_structure(sample, raw_stage)
        stage_traces = _structure_traces(stage_sample, prefix="")
        payloads.append(
            {
                "label": str(np.asarray(stage_sample.get("stage_label", "final")).reshape(-1)[0]),
                "stage_index": int(np.asarray(stage_sample.get("stage_index", stage_index)).reshape(-1)[0]),
                "scheduler_step": int(np.asarray(stage_sample.get("scheduler_step", -1)).reshape(-1)[0]),
                "tau": float(np.asarray(stage_sample.get("tau", 0.0)).reshape(-1)[0]),
                "structure": {
                    "bricks": _serialize_traces(stage_traces["bricks"]),
                    "surfaces": _serialize_traces(stage_traces["surfaces"]),
                    "dipole_specs": stage_traces["dipole_specs"],
                },
                "scores": evaluate_sample_scores(stage_sample),
            }
        )
    return payloads


def _serialize_traces(traces: Sequence[go.BaseTraceType]) -> List[Dict]:
    return [trace.to_plotly_json() for trace in traces]


def _sample_state(sample: Dict, projection: str) -> Dict:
    sampled_trajectory = _trajectory_state_payloads(sample)
    sampled = sampled_trajectory[-1]["structure"]
    original = _structure_traces(sample, prefix="original_") if "original_brick_anchors" in sample else sampled
    target = _target_trace_sets(sample)
    return {
        "meta": _sample_meta(sample),
        "scores": evaluate_sample_scores(sample),
        "sampled_trajectory": sampled_trajectory,
        "scene": _scene_spec(sample, projection=projection),
        "structures": {
            "sampled": sampled,
            "original": {
                "bricks": _serialize_traces(original["bricks"]),
                "surfaces": _serialize_traces(original["surfaces"]),
                "dipole_specs": original["dipole_specs"],
            },
        },
        "target": {key: _serialize_traces(value) for key, value in target.items()},
    }


def _build_html(
    samples: Sequence[Dict],
    initial_index: int,
    structure_view: str,
    display_view: str,
    target_view: str,
    projection: str,
    show_dipoles: bool,
) -> str:
    states = [_sample_state(sample, projection=projection) for sample in samples]
    if initial_index < 0 or initial_index >= len(states):
        raise IndexError(f"Requested sample index {initial_index}, but dataset has {len(states)} samples.")

    if target_view == "surface":
        target_view = "filled"

    initial_state = states[initial_index]
    if structure_view == "original" and not bool(initial_state["meta"]["has_original"]):
        structure_view = "sampled"

    base_layout = {
        "title": {
            "text": "LEGO SH / Dipole Visualizer",
            "x": 0.02,
            "xanchor": "left",
        },
        "paper_bgcolor": "#f4f0e8",
        "plot_bgcolor": "#f4f0e8",
        "margin": {"l": 0, "r": 0, "b": 0, "t": 56},
        "legend": {
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.82)",
            "groupclick": "toggleitem",
            "itemdoubleclick": False,
            "uirevision": "legend",
        },
        "scene": initial_state["scene"],
        "uirevision": f"sample-{initial_index}",
    }
    left_figure = go.Figure(data=[], layout=base_layout)
    right_figure = go.Figure(data=[], layout=base_layout)
    left_plot_html = pio.to_html(
        left_figure,
        full_html=False,
        include_plotlyjs="cdn",
        div_id="lego-plot-left",
        config={"responsive": True, "displaylogo": False},
    )
    right_plot_html = pio.to_html(
        right_figure,
        full_html=False,
        include_plotlyjs=False,
        div_id="lego-plot-right",
        config={"responsive": True, "displaylogo": False},
    )

    state_json = json.dumps(states, cls=PlotlyJSONEncoder)
    layout_json = json.dumps(base_layout, cls=PlotlyJSONEncoder)
    preferred_left_structure = json.dumps(structure_view)
    sample_options = "\n".join(
        f'<option value="{idx}"{" selected" if idx == initial_index else ""}>Sample {idx}</option>'
        for idx in range(len(samples))
    )
    display_selected = {"bricks": "", "surfaces": ""}
    display_selected[display_view] = " selected"
    target_selected = {"hidden": "", "wireframe": "", "filled": "", "voxels": ""}
    target_selected[target_view] = " selected"
    projection_selected = {"orthographic": "", "perspective": ""}
    projection_selected[projection] = " selected"
    dipole_checked = " checked" if show_dipoles else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LEGO SH / Dipole Visualizer</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(237, 229, 209, 0.9), rgba(244, 240, 232, 0.95) 48%),
        linear-gradient(135deg, #f6f0e7, #efe5d2);
      color: #1d2329;
    }}
    .shell {{
      padding: 16px 18px 18px 18px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 12px;
      align-items: end;
      margin-bottom: 10px;
    }}
    .control {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }}
    .control.hidden {{
      display: none;
    }}
    .control label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #58626c;
    }}
    .control select,
    .control input {{
      font: inherit;
      border: 1px solid rgba(29, 35, 41, 0.16);
      background: rgba(255, 255, 255, 0.86);
      border-radius: 10px;
      padding: 9px 10px;
      color: #1d2329;
    }}
    .control.checkbox {{
      flex-direction: row;
      align-items: center;
      gap: 10px;
      margin-top: 20px;
    }}
    .control.checkbox label {{
      margin: 0;
      text-transform: none;
      letter-spacing: 0;
      font-size: 14px;
      color: #1d2329;
    }}
    .slider-value {{
      font-size: 12px;
      color: #58626c;
      text-align: right;
    }}
    .meta {{
      margin: 0 0 12px 0;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(29, 35, 41, 0.08);
      line-height: 1.45;
    }}
    .note {{
      font-size: 13px;
      color: #4e5a65;
      margin-top: 4px;
    }}
    .score-panel {{
      margin: 0 0 12px 0;
      padding: 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.76);
      border: 1px solid rgba(29, 35, 41, 0.08);
    }}
    .score-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      margin-bottom: 12px;
    }}
    .score-table th,
    .score-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid rgba(29, 35, 41, 0.08);
      text-align: left;
      vertical-align: middle;
    }}
    .score-table th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #58626c;
    }}
    .score-name {{
      font-weight: 600;
      color: #1d2329;
    }}
    .score-value {{
      font-variant-numeric: tabular-nums;
      font-weight: 600;
    }}
    .score-value.good {{
      color: #215c33;
    }}
    .score-value.warning {{
      color: #8a6518;
    }}
    .score-value.bad {{
      color: #9a2b2b;
    }}
    .score-delta.negative {{
      color: #9a2b2b;
    }}
    .score-delta.positive {{
      color: #215c33;
    }}
    .score-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .badge-card {{
      display: flex;
      align-items: center;
      gap: 10px;
      border-radius: 999px;
      background: rgba(245, 241, 233, 0.9);
      border: 1px solid rgba(29, 35, 41, 0.08);
      padding: 8px 12px;
    }}
    .badge-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #58626c;
    }}
    .badge-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 72px;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #ffffff;
    }}
    .badge-pill.good {{
      background: #215c33;
    }}
    .badge-pill.warning {{
      background: #8a6518;
    }}
    .badge-pill.bad {{
      background: #9a2b2b;
    }}
    .badge-pill.neutral {{
      background: #697582;
    }}
    .badge-summary {{
      font-size: 13px;
      color: #26303a;
      font-variant-numeric: tabular-nums;
    }}
    .score-details {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .score-card {{
      border-radius: 14px;
      background: rgba(245, 241, 233, 0.9);
      border: 1px solid rgba(29, 35, 41, 0.08);
      padding: 10px 12px;
    }}
    .score-card summary {{
      cursor: pointer;
      font-weight: 600;
      color: #1d2329;
      list-style: none;
    }}
    .score-card summary::-webkit-details-marker {{
      display: none;
    }}
    .score-card-body {{
      margin-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .score-section-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #58626c;
      margin-top: 4px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      line-height: 1.35;
      color: #26303a;
    }}
    .metric-row span:last-child {{
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 12px;
    }}
    .plot-panel {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-width: 0;
    }}
    .plot-panel-title {{
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #1d2329;
      padding: 0 2px;
    }}
    #lego-plot-left,
    #lego-plot-right {{
      height: calc(100vh - 220px);
      min-height: 680px;
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(32, 34, 36, 0.10);
      background: rgba(255, 255, 255, 0.58);
    }}
    .button-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .button-row button {{
      font: inherit;
      border: 1px solid rgba(29, 35, 41, 0.16);
      background: rgba(255, 255, 255, 0.86);
      border-radius: 10px;
      padding: 9px 10px;
      color: #1d2329;
      cursor: pointer;
    }}
    .button-row button:hover {{
      background: rgba(255, 255, 255, 0.98);
    }}
    @media (max-width: 1100px) {{
      .controls {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .score-details {{
        grid-template-columns: repeat(1, minmax(0, 1fr));
      }}
      .plot-grid {{
        grid-template-columns: 1fr;
      }}
      #lego-plot-left,
      #lego-plot-right {{
        min-height: 560px;
        height: calc(100vh - 280px);
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="controls">
      <div class="control">
        <label for="sample-select">Sample</label>
        <select id="sample-select">{sample_options}</select>
      </div>
      <div id="trajectory-control" class="control hidden">
        <label for="trajectory-slider">Sampling Step</label>
        <input id="trajectory-slider" type="range" min="0" max="0" step="1" value="0">
        <div id="trajectory-label" class="slider-value">final</div>
      </div>
      <div class="control">
        <label for="display-select">Geometry</label>
        <select id="display-select">
          <option value="bricks"{display_selected["bricks"]}>Full bricks</option>
          <option value="surfaces"{display_selected["surfaces"]}>Node SH surfaces</option>
        </select>
      </div>
      <div class="control">
        <label for="target-select">Target</label>
        <select id="target-select">
          <option value="hidden"{target_selected["hidden"]}>Hidden</option>
          <option value="wireframe"{target_selected["wireframe"]}>Wireframe</option>
          <option value="filled"{target_selected["filled"]}>Filled</option>
          <option value="voxels"{target_selected["voxels"]}>Voxels</option>
        </select>
      </div>
      <div class="control">
        <label for="projection-select">Projection</label>
        <select id="projection-select">
          <option value="orthographic"{projection_selected["orthographic"]}>Orthographic</option>
          <option value="perspective"{projection_selected["perspective"]}>Perspective</option>
        </select>
      </div>
      <div class="control checkbox">
        <input id="dipole-toggle" type="checkbox"{dipole_checked}>
        <label for="dipole-toggle">Show dipole axes</label>
      </div>
      <div class="control">
        <label for="dipole-length">Dipole length</label>
        <input id="dipole-length" type="range" min="{MIN_DIPOLE_LENGTH}" max="{MAX_DIPOLE_LENGTH}" step="0.05" value="{DEFAULT_DIPOLE_LENGTH}">
        <div id="dipole-length-value" class="slider-value">{DEFAULT_DIPOLE_LENGTH:.2f}</div>
      </div>
      <div class="control">
        <label for="block-opacity">Block opacity</label>
        <input id="block-opacity" type="range" min="{MIN_BLOCK_OPACITY}" max="{MAX_BLOCK_OPACITY}" step="0.05" value="{DEFAULT_BLOCK_OPACITY}">
        <div id="block-opacity-value" class="slider-value">{DEFAULT_BLOCK_OPACITY:.2f}</div>
      </div>
      <div class="control">
        <label>Camera Sync</label>
        <div class="button-row">
          <button id="sync-left-to-right" type="button">Left to Right</button>
          <button id="sync-right-to-left" type="button">Right to Left</button>
        </div>
      </div>
    </div>
    <div class="plot-grid">
      <div class="plot-panel">
        <div id="left-view-label" class="plot-panel-title">Left View</div>
        {left_plot_html}
      </div>
      <div class="plot-panel">
        <div id="right-view-label" class="plot-panel-title">Right View</div>
        {right_plot_html}
      </div>
    </div>
    <div id="sample-meta" class="meta"></div>
    <div id="sample-scores" class="score-panel"></div>
  </div>
  <script>
    const legoStates = {state_json};
    const baseLayout = {layout_json};
    const preferredLeftStructure = {preferred_left_structure};
    const plotLeftEl = document.getElementById("lego-plot-left");
    const plotRightEl = document.getElementById("lego-plot-right");
    const leftViewLabel = document.getElementById("left-view-label");
    const rightViewLabel = document.getElementById("right-view-label");
    const sampleSelect = document.getElementById("sample-select");
    const trajectoryControl = document.getElementById("trajectory-control");
    const trajectorySlider = document.getElementById("trajectory-slider");
    const trajectoryLabel = document.getElementById("trajectory-label");
    const displaySelect = document.getElementById("display-select");
    const targetSelect = document.getElementById("target-select");
    const projectionSelect = document.getElementById("projection-select");
    const dipoleToggle = document.getElementById("dipole-toggle");
    const dipoleLength = document.getElementById("dipole-length");
    const dipoleLengthValue = document.getElementById("dipole-length-value");
    const blockOpacity = document.getElementById("block-opacity");
    const blockOpacityValue = document.getElementById("block-opacity-value");
    const syncLeftToRight = document.getElementById("sync-left-to-right");
    const syncRightToLeft = document.getElementById("sync-right-to-left");
    const metaEl = document.getElementById("sample-meta");
    const scoreEl = document.getElementById("sample-scores");
    let lastRenderedSampleIndex = null;

    function deepClone(value) {{
      return JSON.parse(JSON.stringify(value));
    }}

    function normalizeTargetMode(value) {{
      return value === "surface" ? "filled" : value;
    }}

    function mergeInto(target, source) {{
      Object.keys(source || {{}}).forEach((key) => {{
        const value = source[key];
        if (value && typeof value === "object" && !Array.isArray(value)) {{
          const base = target[key];
          target[key] = mergeInto(base && typeof base === "object" && !Array.isArray(base) ? deepClone(base) : {{}}, value);
        }} else {{
          target[key] = value;
        }}
      }});
      return target;
    }}

    function resolveStructurePair(state) {{
      if (!state.meta.has_original) {{
        return {{
          leftKey: "sampled",
          rightKey: "sampled",
          leftLabel: "Sampled",
          rightLabel: "Sampled",
        }};
      }}
      if (preferredLeftStructure === "original") {{
        return {{
          leftKey: "original",
          rightKey: "sampled",
          leftLabel: "Original",
          rightLabel: "Sampled",
        }};
      }}
      return {{
        leftKey: "sampled",
        rightKey: "original",
        leftLabel: "Sampled",
        rightLabel: "Original",
      }};
    }}

    function sampledTrajectory(state) {{
      if (Array.isArray(state.sampled_trajectory) && state.sampled_trajectory.length > 0) {{
        return state.sampled_trajectory;
      }}
      return [{{
        label: "final",
        stage_index: 0,
        scheduler_step: -1,
        tau: 0.0,
        structure: state.structures.sampled,
        scores: state.scores,
      }}];
    }}

    function formatTrajectoryStage(stage, stageIndex, totalStages) {{
      const label = stage?.label || (stageIndex === totalStages - 1 ? "final" : `stage ${{stageIndex}}`);
      const extras = [];
      if (Number.isFinite(Number(stage?.scheduler_step)) && Number(stage.scheduler_step) >= 0) {{
        extras.push(`t=${{Number(stage.scheduler_step)}}`);
      }}
      if (Number.isFinite(Number(stage?.tau))) {{
        extras.push(`tau=${{Number(stage.tau).toFixed(3)}}`);
      }}
      return extras.length > 0 ? `${{label}} · ${{extras.join(" · ")}}` : label;
    }}

    function updateTrajectoryControl(state, resetToFinal = false) {{
      const trajectory = sampledTrajectory(state);
      const hasTrajectory = trajectory.length > 1;
      trajectoryControl.classList.toggle("hidden", !hasTrajectory);
      trajectorySlider.disabled = !hasTrajectory;
      trajectorySlider.min = "0";
      trajectorySlider.max = String(Math.max(trajectory.length - 1, 0));
      if (resetToFinal || trajectorySlider.value === "") {{
        trajectorySlider.value = String(Math.max(trajectory.length - 1, 0));
      }}
      const clampedIndex = Math.min(
        Math.max(Number(trajectorySlider.value || "0"), 0),
        Math.max(trajectory.length - 1, 0)
      );
      trajectorySlider.value = String(clampedIndex);
      trajectoryLabel.textContent = formatTrajectoryStage(trajectory[clampedIndex], clampedIndex, trajectory.length);
      return clampedIndex;
    }}

    function currentSampledStage(state) {{
      const trajectory = sampledTrajectory(state);
      const stageIndex = Math.min(
        Math.max(Number(trajectorySlider.value || "0"), 0),
        Math.max(trajectory.length - 1, 0)
      );
      return trajectory[stageIndex];
    }}

    function structurePayloadForKey(state, structureKey) {{
      if (structureKey === "sampled") {{
        const stage = currentSampledStage(state);
        if (stage && stage.structure) {{
          return stage.structure;
        }}
      }}
      return state.structures[structureKey];
    }}

    function buildMeta(state, sampleIndex) {{
      const pair = resolveStructurePair(state);
      const sampledStage = currentSampledStage(state);
      const trajectory = sampledTrajectory(state);
      const bits = [
        `Sample ${{sampleIndex}}`,
        `${{state.meta.num_bricks}} bricks`,
        `${{state.meta.num_polar}} polar`,
      ];
      if (state.meta.num_diffused > 0) {{
        bits.push(`${{state.meta.num_diffused}} diffused`);
      }}
      if (Object.prototype.hasOwnProperty.call(state.meta, "source_frame_id")) {{
        bits.push(`source frame ${{state.meta.source_frame_id}}`);
      }}
      if (Object.prototype.hasOwnProperty.call(state.meta, "split_id")) {{
        bits.push(`split ${{state.meta.split_id}}`);
      }}
      if (Object.prototype.hasOwnProperty.call(state.meta, "occupancy_mode")) {{
        bits.push(`${{state.meta.occupancy_mode}} occupancy`);
      }}
      if (Object.prototype.hasOwnProperty.call(state.meta, "sampling_sampler")) {{
        let samplerText = `${{state.meta.sampling_sampler}} sampler`;
        if (Object.prototype.hasOwnProperty.call(state.meta, "sampling_steps")) {{
          samplerText += `, ${{state.meta.sampling_steps}} steps`;
        }}
        if (
          Object.prototype.hasOwnProperty.call(state.meta, "sampling_late_refine_factor") &&
          Number(state.meta.sampling_late_refine_factor) > 1 &&
          Object.prototype.hasOwnProperty.call(state.meta, "sampling_late_refine_from_step") &&
          Number(state.meta.sampling_late_refine_from_step) >= 0
        ) {{
          samplerText += `, refine<=${{state.meta.sampling_late_refine_from_step}} x${{state.meta.sampling_late_refine_factor}}`;
        }}
        if (
          Object.prototype.hasOwnProperty.call(state.meta, "sampling_linger_count") &&
          Number(state.meta.sampling_linger_count) > 0 &&
          Object.prototype.hasOwnProperty.call(state.meta, "sampling_linger_step") &&
          Number(state.meta.sampling_linger_step) >= 0
        ) {{
          samplerText += `, linger@${{state.meta.sampling_linger_step}} +${{state.meta.sampling_linger_count}}`;
        }}
        if (Object.prototype.hasOwnProperty.call(state.meta, "sampling_clash_guidance")) {{
          if (state.meta.sampling_clash_guidance) {{
            let guidanceText = "clash guidance on";
            if (Object.prototype.hasOwnProperty.call(state.meta, "sampling_clash_guidance_strength")) {{
              guidanceText += `, s=${{Number(state.meta.sampling_clash_guidance_strength).toFixed(2)}}`;
            }}
            if (Object.prototype.hasOwnProperty.call(state.meta, "sampling_clash_guidance_weight_schedule")) {{
              guidanceText += `, ${{state.meta.sampling_clash_guidance_weight_schedule}}`;
            }}
            if (Object.prototype.hasOwnProperty.call(state.meta, "sampling_clash_guidance_auto_scale")) {{
              guidanceText += state.meta.sampling_clash_guidance_auto_scale ? ", auto-scale" : ", fixed-scale";
            }}
            if (Object.prototype.hasOwnProperty.call(state.meta, "sampling_cohesion_guidance_strength")) {{
              const cgs = Number(state.meta.sampling_cohesion_guidance_strength);
              if (cgs > 0) {{
                guidanceText += `, cohesion=${{cgs.toFixed(2)}}`;
                if (Object.prototype.hasOwnProperty.call(state.meta, "sampling_cohesion_guidance_target_contacts")) {{
                  guidanceText += `@${{Number(state.meta.sampling_cohesion_guidance_target_contacts).toFixed(1)}}`;
                }}
              }}
            }}
            bits.push(guidanceText);
          }} else {{
            bits.push("clash guidance off");
          }}
        }}
        bits.push(samplerText);
      }}
      if (trajectory.length > 1) {{
        bits.push(`stage ${{formatTrajectoryStage(sampledStage, Number(trajectorySlider.value || "0"), trajectory.length)}}`);
      }}
      if (Object.prototype.hasOwnProperty.call(state.meta, "trajectory_stride") && Number(state.meta.trajectory_stride) > 1) {{
        bits.push(`trajectory stride ${{state.meta.trajectory_stride}}`);
      }}
      const shellDetails = [];
      if (state.meta.occupancy_mode === "shell") {{
        if (Object.prototype.hasOwnProperty.call(state.meta, "shell_thickness")) {{
          shellDetails.push(`thickness ${{Number(state.meta.shell_thickness).toFixed(2)}}`);
        }}
        if (Object.prototype.hasOwnProperty.call(state.meta, "shell_sparsity")) {{
          shellDetails.push(`sparsity ${{Number(state.meta.shell_sparsity).toFixed(2)}}`);
        }}
      }}
      const structureLabel = state.meta.has_original
        ? `${{pair.leftLabel}} is shown on the left and ${{pair.rightLabel.toLowerCase()}} on the right. Rotate them independently, use the sync buttons above, or double-click a view to copy its camera to the other.`
        : "Both views show the same sampled assembly because this dataset has no original/reference structure.";
      metaEl.innerHTML = `
        <strong>${{bits.join(" · ")}}</strong>
        <div class="note">
          Smooth SH target = the continuous spherical-harmonic surface before voxelization and brick approximation.
          Geometry view: Full bricks renders brick type/rotation geometry. On sampled trajectories, ligand brick type/rotation is decoded from the current SH shape field at each stage; Node SH surfaces shows that diffused per-node SH shape field directly.
          ${{shellDetails.length > 0 ? ` Shell generation: ${{shellDetails.join(" · ")}}.` : ""}}
          ${{structureLabel}}
        </div>
      `;
    }}

    function formatMetric(value, digits = 3) {{
      if (value === null || value === undefined) {{
        return "n/a";
      }}
      if (typeof value === "boolean") {{
        return value ? "yes" : "no";
      }}
      if (typeof value === "number") {{
        if (!Number.isFinite(value)) {{
          return "n/a";
        }}
        return Number(value).toFixed(digits);
      }}
      return String(value);
    }}

    function formatDelta(value) {{
      if (value === null || value === undefined || !Number.isFinite(Number(value))) {{
        return {{ text: "n/a", className: "" }};
      }}
      const numeric = Number(value);
      const sign = numeric > 1e-8 ? "+" : "";
      return {{
        text: `${{sign}}${{numeric.toFixed(2)}}`,
        className: numeric < -1e-8 ? "negative" : numeric > 1e-8 ? "positive" : "",
      }};
    }}

    function scoreTone(value) {{
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {{
        return "";
      }}
      if (numeric >= 85.0) {{
        return "good";
      }}
      if (numeric >= 60.0) {{
        return "warning";
      }}
      return "bad";
    }}

    function overallScore(scoreSet) {{
      if (!scoreSet) {{
        return NaN;
      }}
      const validity = Number(scoreSet.validity);
      const dipoles = Number(scoreSet.dipoles);
      if (![validity, dipoles].every(Number.isFinite)) {{
        return NaN;
      }}
      return 0.70 * validity + 0.30 * dipoles;
    }}

    function badgePayload(scoreSet) {{
      const overall = overallScore(scoreSet);
      const validity = Number(scoreSet?.validity);
      if (!Number.isFinite(overall) || !Number.isFinite(validity)) {{
        return {{
          label: "n/a",
          className: "neutral",
          summary: "overall n/a",
        }};
      }}
      if (validity < 5.0) {{
        return {{
          label: "Fail",
          className: "bad",
          summary: `overall ${{overall.toFixed(1)}}`,
        }};
      }}
      if (validity >= 95.0 && overall >= 80.0) {{
        return {{
          label: "Pass",
          className: "good",
          summary: `overall ${{overall.toFixed(1)}}`,
        }};
      }}
      if (validity >= 60.0 && overall >= 55.0) {{
        return {{
          label: "Warning",
          className: "warning",
          summary: `overall ${{overall.toFixed(1)}}`,
        }};
      }}
      return {{
        label: "Fail",
        className: "bad",
        summary: `overall ${{overall.toFixed(1)}}`,
      }};
    }}

    function renderBadgeCard(label, badge) {{
      return `
        <div class="badge-card">
          <span class="badge-label">${{label}}</span>
          <span class="badge-pill ${{badge.className}}">${{badge.label}}</span>
          <span class="badge-summary">${{badge.summary}}</span>
        </div>
      `;
    }}

    function renderMetricRows(rows, metrics) {{
      return rows.map(([label, key, digits]) => {{
        const value = Object.prototype.hasOwnProperty.call(metrics, key) ? metrics[key] : null;
        return `
          <div class="metric-row">
            <span>${{label}}</span>
            <span>${{formatMetric(value, digits)}}</span>
          </div>
        `;
      }}).join("");
    }}

    function buildScoreCardDetails(title, payload, openByDefault = true) {{
      if (!payload) {{
        return "";
      }}
      const metrics = payload.metrics || {{}};
      const hasTargetMetrics = Object.prototype.hasOwnProperty.call(metrics, "target_f1");
      const targetSection = hasTargetMetrics
        ? `
          <div class="score-section-title">Target</div>
          ${{renderMetricRows([
            ["Target coverage", "target_coverage", 3],
            ["Target precision", "target_precision", 3],
            ["Target F1", "target_f1", 3],
            ["Mean target->structure", "mean_target_to_structure_distance", 3],
            ["Mean structure->target", "mean_structure_to_target_distance", 3],
          ], metrics)}}
        `
        : "";
      return `
        <details class="score-card"${{openByDefault ? " open" : ""}}>
          <summary>${{title}}</summary>
          <div class="score-card-body">
            <div class="score-section-title">Validity</div>
            ${{renderMetricRows([
              ["Valid-like geometry", "is_valid_like", 0],
              ["Overlap volume", "total_overlap_volume", 3],
              ["Effective overlap", "effective_overlap_volume", 3],
              ["Clashing brick pairs", "clashing_brick_pairs", 0],
              ["Micro overlap pairs", "micro_overlapping_pairs", 0],
              ["Severe overlap pairs", "severe_overlapping_pairs", 0],
              ["Max pair overlap", "max_pair_overlap_volume", 3],
              ["Connected components", "num_components", 0],
            ], metrics)}}
            <div class="score-section-title">Compactness</div>
            ${{renderMetricRows([
              ["Matched face area", "matched_face_area", 3],
              ["Matched face ratio", "matched_face_ratio", 3],
              ["Connected brick pairs", "connected_brick_pairs", 0],
              ["Intrinsic faces", "intrinsic_face_count", 0],
            ], metrics)}}
            <div class="score-section-title">Shellness</div>
            ${{renderMetricRows([
              ["Shell surface ratio", "shell_surface_ratio", 3],
            ], metrics)}}
            <div class="score-section-title">Dipoles</div>
            ${{renderMetricRows([
              ["Attractive contact area", "attractive_contact_area", 3],
              ["Repulsive contact area", "repulsive_contact_area", 3],
              ["Neutral contact area", "neutral_contact_area", 3],
              ["Total contact area", "total_contact_area", 3],
              ["Weighted dipole energy", "weighted_dipole_energy", 3],
              ["Mean weighted energy", "mean_weighted_dipole_energy", 3],
            ], metrics)}}
            ${{targetSection}}
          </div>
        </details>
      `;
    }}

    function buildCompareDetails(comparePayload) {{
      if (!comparePayload) {{
        return "";
      }}
      const shift = comparePayload.anchor_shift || {{}};
      const deltas = comparePayload.metric_delta || {{}};
      const scoreDelta = comparePayload.score_delta || {{}};
      return `
        <details class="score-card">
          <summary>Sampled vs Original</summary>
          <div class="score-card-body">
            <div class="score-section-title">Scores</div>
            <div class="metric-row"><span>Validity delta</span><span>${{formatMetric(scoreDelta.validity, 2)}}</span></div>
            <div class="metric-row"><span>Compactness delta</span><span>${{formatMetric(scoreDelta.compactness, 2)}}</span></div>
            <div class="metric-row"><span>Shellness delta</span><span>${{formatMetric(scoreDelta.shellness, 2)}}</span></div>
            <div class="metric-row"><span>Dipoles delta</span><span>${{formatMetric(scoreDelta.dipoles, 2)}}</span></div>
            <div class="score-section-title">Geometry</div>
            ${{renderMetricRows([
              ["Diffused shift mean", "diffused_shift_mean", 3],
              ["Diffused shift max", "diffused_shift_max", 3],
              ["Fixed shift mean", "fixed_shift_mean", 3],
              ["Fixed shift max", "fixed_shift_max", 3],
            ], shift)}}
            <div class="score-section-title">Metric deltas</div>
            ${{renderMetricRows([
              ["Overlap delta", "overlap_volume", 3],
              ["Matched-face ratio delta", "matched_face_ratio", 3],
              ["Shell surface delta", "shell_surface_ratio", 3],
              ["Dipole energy delta", "weighted_dipole_energy", 3],
            ], deltas)}}
          </div>
        </details>
      `;
    }}

    function buildScorePanel(state) {{
      const sampledStage = currentSampledStage(state);
      const scorePayload = sampledStage?.scores || state.scores || null;
      if (!scorePayload || !scorePayload.sampled) {{
        scoreEl.innerHTML = "";
        return;
      }}

      const sampledScores = scorePayload.sampled.scores || {{}};
      const originalScores = scorePayload.original ? scorePayload.original.scores || {{}} : null;
      const sampledOverall = overallScore(sampledScores);
      const originalOverall = originalScores ? overallScore(originalScores) : null;
      const overallDelta = formatDelta(
        originalScores && Number.isFinite(sampledOverall) && Number.isFinite(originalOverall)
          ? sampledOverall - originalOverall
          : null
      );
      const validityDelta = formatDelta(scorePayload.compare?.score_delta?.validity);
      const compactnessDelta = formatDelta(scorePayload.compare?.score_delta?.compactness);
      const shellnessDelta = formatDelta(scorePayload.compare?.score_delta?.shellness);
      const dipolesDelta = formatDelta(scorePayload.compare?.score_delta?.dipoles);
      const sampledBadge = badgePayload(sampledScores);
      const originalBadge = originalScores ? badgePayload(originalScores) : null;

      scoreEl.innerHTML = `
        <div class="score-badges">
          ${{renderBadgeCard("Sampled", sampledBadge)}}
          ${{originalBadge ? renderBadgeCard("Original", originalBadge) : ""}}
        </div>
        <table class="score-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Sampled</th>
              <th>Original</th>
              <th>Delta</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="score-name">Overall</td>
              <td class="score-value ${{scoreTone(sampledOverall)}}">${{formatMetric(sampledOverall, 2)}}</td>
              <td class="score-value ${{originalScores ? scoreTone(originalOverall) : ""}}">${{originalScores ? formatMetric(originalOverall, 2) : "n/a"}}</td>
              <td class="score-value score-delta ${{overallDelta.className}}">${{overallDelta.text}}</td>
            </tr>
            <tr>
              <td class="score-name">Validity</td>
              <td class="score-value ${{scoreTone(sampledScores.validity)}}">${{formatMetric(sampledScores.validity, 2)}}</td>
              <td class="score-value ${{originalScores ? scoreTone(originalScores.validity) : ""}}">${{originalScores ? formatMetric(originalScores.validity, 2) : "n/a"}}</td>
              <td class="score-value score-delta ${{validityDelta.className}}">${{validityDelta.text}}</td>
            </tr>
            <tr>
              <td class="score-name">Compactness</td>
              <td class="score-value ${{scoreTone(sampledScores.compactness)}}">${{formatMetric(sampledScores.compactness, 2)}}</td>
              <td class="score-value ${{originalScores ? scoreTone(originalScores.compactness) : ""}}">${{originalScores ? formatMetric(originalScores.compactness, 2) : "n/a"}}</td>
              <td class="score-value score-delta ${{compactnessDelta.className}}">${{compactnessDelta.text}}</td>
            </tr>
            <tr>
              <td class="score-name">Shellness</td>
              <td class="score-value ${{scoreTone(sampledScores.shellness)}}">${{formatMetric(sampledScores.shellness, 2)}}</td>
              <td class="score-value ${{originalScores ? scoreTone(originalScores.shellness) : ""}}">${{originalScores ? formatMetric(originalScores.shellness, 2) : "n/a"}}</td>
              <td class="score-value score-delta ${{shellnessDelta.className}}">${{shellnessDelta.text}}</td>
            </tr>
            <tr>
              <td class="score-name">Dipoles</td>
              <td class="score-value ${{scoreTone(sampledScores.dipoles)}}">${{formatMetric(sampledScores.dipoles, 2)}}</td>
              <td class="score-value ${{originalScores ? scoreTone(originalScores.dipoles) : ""}}">${{originalScores ? formatMetric(originalScores.dipoles, 2) : "n/a"}}</td>
              <td class="score-value score-delta ${{dipolesDelta.className}}">${{dipolesDelta.text}}</td>
            </tr>
          </tbody>
        </table>
        <div class="score-details">
          ${{buildScoreCardDetails("Sampled details", scorePayload.sampled, false)}}
          ${{scorePayload.original ? buildScoreCardDetails("Original details", scorePayload.original, false) : ""}}
          ${{buildCompareDetails(scorePayload.compare)}}
        </div>
      `;
    }}

    function applyStructureOpacity(traces) {{
      const opacity = Number(blockOpacity.value);
      return traces.map((trace) => {{
        if (trace.type === "mesh3d" || trace.type === "surface") {{
          trace.opacity = opacity;
        }}
        return trace;
      }});
    }}

    function sceneifyTrace(trace, showLegend, fallbackGroup, fallbackUid) {{
      trace.scene = "scene";
      trace.showlegend = Boolean(showLegend) && trace.showlegend !== false;
      trace.legendgroup = trace.legendgroup || fallbackGroup;
      trace.uid = `${{trace.uid || fallbackUid}}-scene`;
      return trace;
    }}

    function structureTracesForPlot(state, structureKey, display, showLegend) {{
      const structurePayload = structurePayloadForKey(state, structureKey);
      const traces = applyStructureOpacity(deepClone(structurePayload[display]));
      return traces.map((trace, index) =>
        sceneifyTrace(
          trace,
          showLegend,
          trace.legendgroup || `brick-pair-${{structureKey}}-${{index}}`,
          trace.uid || `${{structureKey}}-${{display}}-${{index}}`
        )
      );
    }}

    function targetTracesForPlot(state, targetMode, showLegend) {{
      const traces = deepClone(state.target[targetMode] || []);
      const legendGroup = `target-${{targetMode}}`;
      return traces.map((trace, index) =>
        sceneifyTrace(
          trace,
          showLegend && index === 0,
          legendGroup,
          `target-${{targetMode}}-${{index}}`
        )
      );
    }}

    function buildDipoleTraces(specs) {{
      const lengthScale = Number(dipoleLength.value);
      const coneRef = Math.max(0.12, 0.55 * Math.max(0.12, 0.28 * lengthScale));
      const traces = [];
      specs.forEach((spec, specIndex) => {{
        const lineX = [];
        const lineY = [];
        const lineZ = [];
        const coneX = [];
        const coneY = [];
        const coneZ = [];
        const coneU = [];
        const coneV = [];
        const coneW = [];
        spec.centers.forEach((center, idx) => {{
          const direction = spec.directions[idx];
          const strength = Number(spec.strengths[idx]);
          const arrowLength = Math.max(0.0, lengthScale * strength);
          if (arrowLength <= 1e-8) {{
            return;
          }}
          const coneLength = Math.min(Math.max(0.12, 0.28 * lengthScale), 0.65 * arrowLength);
          const tip = center.map((value, axis) => value + 0.5 * arrowLength * direction[axis]);
          const start = center.map((value, axis) => value - 0.5 * arrowLength * direction[axis]);
          const shaftEnd = tip.map((value, axis) => value - coneLength * direction[axis]);
          lineX.push(start[0], shaftEnd[0], null);
          lineY.push(start[1], shaftEnd[1], null);
          lineZ.push(start[2], shaftEnd[2], null);
          coneX.push(tip[0]);
          coneY.push(tip[1]);
          coneZ.push(tip[2]);
          coneU.push(coneLength * direction[0]);
          coneV.push(coneLength * direction[1]);
          coneW.push(coneLength * direction[2]);
        }});
        if (coneX.length === 0) {{
          return;
        }}
        traces.push({{
          type: "scatter3d",
          x: lineX,
          y: lineY,
          z: lineZ,
          scene: "scene",
          mode: "lines",
          line: {{ width: 7, color: spec.color }},
          name: spec.name,
          showlegend: false,
          hoverinfo: "skip",
          uid: `dipole-line-${{specIndex}}`
        }});
        traces.push({{
          type: "cone",
          x: coneX,
          y: coneY,
          z: coneZ,
          u: coneU,
          v: coneV,
          w: coneW,
          scene: "scene",
          anchor: "tip",
          showscale: false,
          showlegend: false,
          hoverinfo: "skip",
          colorscale: [[0.0, spec.color], [1.0, spec.color]],
          cmin: 0.0,
          cmax: 1.0,
          sizemode: "absolute",
          sizeref: coneRef,
          name: spec.name,
          uid: `dipole-cone-${{specIndex}}`
        }});
      }});
      return traces;
    }}

    function currentTracesForStructure(state, structureKey) {{
      const display = displaySelect.value;
      const targetMode = normalizeTargetMode(targetSelect.value);
      let traces = [];
      traces = traces.concat(structureTracesForPlot(state, structureKey, display, true));
      if (dipoleToggle.checked) {{
        traces = traces.concat(buildDipoleTraces(structurePayloadForKey(state, structureKey).dipole_specs));
      }}
      traces = traces.concat(targetTracesForPlot(state, targetMode, true));
      return traces;
    }}

    function cameraWithProjection(camera) {{
      const result = deepClone(camera || {{}});
      result.projection = {{ type: projectionSelect.value }};
      return result;
    }}

    function getSceneCamera(plotEl) {{
      const fullScene = plotEl && plotEl._fullLayout && plotEl._fullLayout.scene;
      if (fullScene && fullScene.camera) {{
        return deepClone(fullScene.camera);
      }}
      const layoutScene = plotEl && plotEl.layout && plotEl.layout.scene;
      if (layoutScene && layoutScene.camera) {{
        return deepClone(layoutScene.camera);
      }}
      return null;
    }}

    function currentLayout(state, sampleIndex, plotEl) {{
      const layout = deepClone(baseLayout);
      const sceneBase = mergeInto(deepClone(layout.scene || {{}}), deepClone(state.scene));
      sceneBase.camera = cameraWithProjection(sceneBase.camera || {{}});

      if (lastRenderedSampleIndex === sampleIndex) {{
        const currentCamera = getSceneCamera(plotEl);
        if (currentCamera) {{
          sceneBase.camera = cameraWithProjection(currentCamera);
        }}
      }}

      layout.scene = sceneBase;
      layout.scene.uirevision = `scene-${{sampleIndex}}-${{projectionSelect.value}}`;
      layout.legend = layout.legend || {{}};
      layout.legend.uirevision = `legend-${{sampleIndex}}`;
      layout.uirevision = `sample-${{sampleIndex}}`;
      layout.annotations = [];
      return layout;
    }}

    function normalizeCamera(camera) {{
      if (!camera || typeof camera !== "object") {{
        return null;
      }}
      return {{
        center: camera.center || null,
        eye: camera.eye || null,
        up: camera.up || null,
        projection: camera.projection || null,
      }};
    }}

    function cameraEquals(left, right) {{
      return JSON.stringify(normalizeCamera(left)) === JSON.stringify(normalizeCamera(right));
    }}

    function syncCamera(sourcePlotEl, targetPlotEl) {{
      const sourceCamera = getSceneCamera(sourcePlotEl);
      if (!sourceCamera) {{
        return;
      }}
      const nextCamera = cameraWithProjection(sourceCamera);
      const currentCamera = getSceneCamera(targetPlotEl) || {{}};
      if (cameraEquals(currentCamera, nextCamera)) {{
        return;
      }}
      Plotly.relayout(targetPlotEl, {{ "scene.camera": nextCamera }});
    }}

    function viewPair(state) {{
      const pair = resolveStructurePair(state);
      return {{
        left: pair.leftKey,
        right: pair.rightKey,
        leftLabel: pair.leftLabel,
        rightLabel: pair.rightLabel,
      }};
    }}

    function render() {{
      const sampleIndex = Number(sampleSelect.value);
      const state = legoStates[sampleIndex];
      const sampleChanged = lastRenderedSampleIndex !== sampleIndex;
      const pair = viewPair(state);
      updateTrajectoryControl(state, sampleChanged);
      leftViewLabel.textContent = pair.leftLabel;
      rightViewLabel.textContent = pair.rightLabel;
      const leftTraces = currentTracesForStructure(state, pair.left);
      const rightTraces = currentTracesForStructure(state, pair.right);
      const leftLayout = currentLayout(state, sampleIndex, plotLeftEl);
      const rightLayout = currentLayout(state, sampleIndex, plotRightEl);
      Plotly.react(plotLeftEl, leftTraces, leftLayout, {{ responsive: true, displaylogo: false }});
      Plotly.react(plotRightEl, rightTraces, rightLayout, {{ responsive: true, displaylogo: false }});
      lastRenderedSampleIndex = sampleIndex;
      dipoleLengthValue.textContent = Number(dipoleLength.value).toFixed(2);
      blockOpacityValue.textContent = Number(blockOpacity.value).toFixed(2);
      buildMeta(state, sampleIndex);
      buildScorePanel(state);
    }}

    function mirrorLegendVisibility(sourcePlotEl, targetPlotEl, eventData) {{
      const clickedTrace = sourcePlotEl.data[eventData.curveNumber];
      if (!clickedTrace) {{
        return;
      }}
      const legendGroup = clickedTrace.legendgroup || clickedTrace.uid || `trace-${{eventData.curveNumber}}`;
      window.setTimeout(() => {{
        const sourceTraceAfter = sourcePlotEl.data[eventData.curveNumber];
        if (!sourceTraceAfter) {{
          return;
        }}
        const sourceVisibility = sourceTraceAfter.visible;
        const nextVisibility = sourceVisibility === "legendonly" ? "legendonly" : true;
        const traceIndices = [];
        targetPlotEl.data.forEach((trace, traceIndex) => {{
          const candidateGroup = trace.legendgroup || trace.uid || `trace-${{traceIndex}}`;
          if (candidateGroup === legendGroup) {{
            traceIndices.push(traceIndex);
          }}
        }});
        if (traceIndices.length > 0) {{
          Plotly.restyle(targetPlotEl, {{ visible: nextVisibility }}, traceIndices);
        }}
      }}, 0);
    }}

    plotLeftEl.on("plotly_legendclick", (eventData) => {{
      mirrorLegendVisibility(plotLeftEl, plotRightEl, eventData);
    }});
    plotRightEl.on("plotly_legendclick", (eventData) => {{
      mirrorLegendVisibility(plotRightEl, plotLeftEl, eventData);
    }});
    plotLeftEl.on("plotly_doubleclick", () => {{
      window.setTimeout(() => syncCamera(plotLeftEl, plotRightEl), 0);
      return false;
    }});
    plotRightEl.on("plotly_doubleclick", () => {{
      window.setTimeout(() => syncCamera(plotRightEl, plotLeftEl), 0);
      return false;
    }});

    [sampleSelect, displaySelect, targetSelect, projectionSelect, dipoleToggle].forEach((element) => {{
      element.addEventListener("change", render);
    }});
    trajectorySlider.addEventListener("input", render);
    dipoleLength.addEventListener("input", render);
    blockOpacity.addEventListener("input", render);
    syncLeftToRight.addEventListener("click", () => syncCamera(plotLeftEl, plotRightEl));
    syncRightToLeft.addEventListener("click", () => syncCamera(plotRightEl, plotLeftEl));

    render();
  </script>
</body>
</html>
"""


def _resolve_output_html(path: Path, output_html: Path | None) -> Path:
    if output_html is not None:
        return output_html
    with tempfile.NamedTemporaryFile(
        prefix=f"{path.stem}_viewer_",
        suffix=".html",
        delete=False,
    ) as handle:
        return Path(handle.name)


def main() -> None:
    args = parse_args()
    if args.trajectory_stride < 1:
        raise ValueError("--trajectory-stride must be >= 1.")

    samples = load_samples(args.path)
    if args.trajectory_stride > 1:
        samples = [_apply_trajectory_stride(sample, args.trajectory_stride) for sample in samples]
    if len(samples) == 0:
        raise ValueError(f"No samples found in {args.path}.")

    initial_target = "voxels" if args.show_target_voxels else args.target_view
    if initial_target == "surface":
        initial_target = "filled"
    show_dipoles = bool(args.show_dipoles or args.show_ports)

    html = _build_html(
        samples=samples,
        initial_index=int(args.index),
        structure_view=args.structure_view,
        display_view=args.display_view,
        target_view=initial_target,
        projection=args.projection,
        show_dipoles=show_dipoles,
    )
    output_html = _resolve_output_html(args.path, args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    if args.output_html is not None:
        print(f"Saved visualization to {output_html}")
    elif args.no_show:
        print(f"Built temporary visualization at {output_html}")

    if not args.no_show:
        webbrowser.open(output_html.resolve().as_uri(), new=2)


if __name__ == "__main__":
    main()
