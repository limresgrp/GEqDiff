from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from lego.lego_blocks import LEGO_LIBRARY, rotated_offsets
    from lego.lego_visualizer import _mesh_trace_from_brick, _scene_spec, _surface_trace_from_brick
except ModuleNotFoundError:
    from lego_blocks import LEGO_LIBRARY, rotated_offsets
    from lego_visualizer import _mesh_trace_from_brick, _scene_spec, _surface_trace_from_brick


def _valid_nodes(mask: np.ndarray) -> np.ndarray:
    return ~np.asarray(mask, dtype=bool)


def _trim_edge_index(edge_index: np.ndarray) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return edge_index.reshape(2, 0)
    keep = edge_index[0] >= 0
    return edge_index[:, keep]


def _connected(mask: np.ndarray, edge_index: np.ndarray) -> bool:
    nodes = np.flatnonzero(mask)
    if nodes.size <= 1:
        return True
    adjacency = {int(node): [] for node in nodes.tolist()}
    for src, dst in _trim_edge_index(edge_index).T.tolist():
        if src in adjacency and dst in adjacency:
            adjacency[src].append(dst)
    start = int(nodes[0])
    stack = [start]
    visited = {start}
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(int(neighbor))
            stack.append(int(neighbor))
    return len(visited) == len(adjacency)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a padded LEGO diffusion dataset NPZ.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the split dataset NPZ.")
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Optional comma-separated example indices to visualize, e.g. '0,4,12'. Blank means all examples.",
    )
    parser.add_argument(
        "--plot-html",
        type=Path,
        default=None,
        help="Optional HTML path to render selected diffusion examples. Default: <input>.html",
    )
    parser.add_argument(
        "--open-html",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the generated HTML in a browser window.",
    )
    parser.add_argument(
        "--designable-label",
        type=str,
        default="Designable set",
        help="Legend label for the masked/diffused subset (legacy ligand).",
    )
    parser.add_argument(
        "--context-label",
        type=str,
        default="Context set",
        help="Legend label for the fixed subset (legacy pocket).",
    )
    parser.add_argument(
        "--initial-projection",
        type=str,
        default="orthographic",
        choices=["orthographic", "perspective"],
        help="Initial projection mode for the HTML viewer.",
    )
    return parser.parse_args()


def _parse_indices_arg(raw: str) -> list[int]:
    raw = str(raw).strip()
    if raw == "":
        return []
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        values.append(int(token))
    return values


def _as_int_rotation(rotation: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(rotation, dtype=np.float32)).astype(np.int32)


def _brick_face_triangles(anchor: np.ndarray, brick_type: str, rotation: np.ndarray) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
    if brick_type not in LEGO_LIBRARY:
        return [], [], [], [], [], []
    offsets = np.asarray(LEGO_LIBRARY[brick_type]["offsets"], dtype=np.int32)
    voxels = rotated_offsets(offsets, _as_int_rotation(rotation)).astype(np.int32) + np.rint(anchor).astype(np.int32)
    occupied = {tuple(v.tolist()) for v in voxels}
    directions = (
        np.asarray((1, 0, 0), dtype=np.int32),
        np.asarray((-1, 0, 0), dtype=np.int32),
        np.asarray((0, 1, 0), dtype=np.int32),
        np.asarray((0, -1, 0), dtype=np.int32),
        np.asarray((0, 0, 1), dtype=np.int32),
        np.asarray((0, 0, -1), dtype=np.int32),
    )

    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []

    def face_vertices(center: np.ndarray, direction: np.ndarray) -> np.ndarray:
        dx, dy, dz = direction.tolist()
        if abs(dx) > 0:
            u = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
            v = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        elif abs(dy) > 0:
            u = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
            v = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        else:
            u = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
            v = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        c = center + 0.5 * direction.astype(np.float32)
        return np.asarray(
            [
                c - 0.5 * u - 0.5 * v,
                c + 0.5 * u - 0.5 * v,
                c + 0.5 * u + 0.5 * v,
                c - 0.5 * u + 0.5 * v,
            ],
            dtype=np.float32,
        )

    for voxel in voxels.astype(np.float32):
        for direction in directions:
            neighbor = tuple((voxel.astype(np.int32) + direction).tolist())
            if neighbor in occupied:
                continue
            verts = face_vertices(voxel, direction)
            start = len(x)
            x.extend(verts[:, 0].tolist())
            y.extend(verts[:, 1].tolist())
            z.extend(verts[:, 2].tolist())
            i.extend([start + 0, start + 0])
            j.extend([start + 1, start + 2])
            k.extend([start + 2, start + 3])
    return x, y, z, i, j, k


def _single_brick_mesh_trace(
    anchor: np.ndarray,
    brick_type: str,
    rotation: np.ndarray,
    color: str,
    name: str,
    legendgroup: str,
) -> go.Mesh3d | None:
    x, y, z, i, j, k = _brick_face_triangles(anchor, brick_type, rotation)
    if len(x) == 0:
        return None
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=1.0,
        flatshading=True,
        name=name,
        legendgroup=legendgroup,
        showlegend=True,
        hovertemplate=f"{name}<extra></extra>",
    )


def _example_traces(
    data: np.lib.npyio.NpzFile,
    example_idx: int,
    designable_label: str,
    context_label: str,
    projection: str,
) -> dict:
    valid_nodes = ~np.asarray(data["pos__mask__"][example_idx], dtype=bool)
    anchors = np.asarray(data["pos"][example_idx], dtype=np.float32)[valid_nodes]
    rotations = np.asarray(data["rotations"][example_idx], dtype=np.float32)[valid_nodes]
    if "types" in data.files:
        types = np.asarray(data["types"][example_idx])[valid_nodes]
    else:
        types = np.asarray(["1x1"] * int(valid_nodes.sum()))
    ligand_mask = np.asarray(data["ligand_mask"][example_idx], dtype=bool)[valid_nodes]

    if "shape_features_raw" in data.files:
        shape_features = np.asarray(data["shape_features_raw"][example_idx], dtype=np.float32)[valid_nodes]
    else:
        shape_scalar = np.asarray(data["shape_scalar_features"][example_idx], dtype=np.float32)[valid_nodes]
        shape_equiv = np.asarray(data["shape_equiv_features"][example_idx], dtype=np.float32)[valid_nodes]
        shape_features = np.concatenate([shape_scalar, shape_equiv], axis=-1)

    if "dipole_direction_raw" in data.files:
        dipole_direction = np.asarray(data["dipole_direction_raw"][example_idx], dtype=np.float32)[valid_nodes]
    else:
        dipole_direction = np.asarray(data["dipole_direction"][example_idx], dtype=np.float32)[valid_nodes]
    if "dipole_strength_raw" in data.files:
        dipole_strength = np.asarray(data["dipole_strength_raw"][example_idx], dtype=np.float32)[valid_nodes]
    else:
        dipole_strength = np.asarray(data["dipole_strength"][example_idx], dtype=np.float32)[valid_nodes]
    dipole_vec = (dipole_direction * dipole_strength).astype(np.float32)

    source_frame_id = int(np.asarray(data["source_frame_id"][example_idx]).reshape(-1)[0]) if "source_frame_id" in data.files else -1
    split_id = int(np.asarray(data["split_id"][example_idx]).reshape(-1)[0]) if "split_id" in data.files else -1
    sample = {
        "brick_anchors": anchors.astype(np.float32),
        "brick_rotations": rotations.astype(np.float32),
        "brick_types": types,
        "brick_features": shape_features.astype(np.float32),
        "brick_dipoles": dipole_vec.astype(np.float32),
        "sampled_brick_mask": ligand_mask.astype(bool),
        "source_frame_id": np.asarray(source_frame_id, dtype=np.int64),
        "split_id": np.asarray(split_id, dtype=np.int64),
    }

    traces = {
        "bricks": {"dipoles": [], "groups": []},
        "surfaces": {"dipoles": [], "groups": []},
    }
    num_nodes = int(anchors.shape[0])
    for node_idx in range(num_nodes):
        brick_type = str(types[node_idx])
        rotation = np.asarray(rotations[node_idx], dtype=np.float32)
        anchor = np.asarray(anchors[node_idx], dtype=np.float32)
        is_designable = bool(ligand_mask[node_idx])
        group_color = "#d46a6a" if is_designable else "#647aa3"
        group_name = designable_label if is_designable else context_label
        brick_name = f"{group_name} {node_idx + 1:02d}: {brick_type}"
        legendgroup = f"brick-{example_idx}-{node_idx}"

        mesh_d = _mesh_trace_from_brick(
            sample=sample,
            anchors=anchors,
            types=types,
            rotations=rotations,
            dipoles=dipole_vec,
            brick_index=node_idx,
        )
        mesh_g = _single_brick_mesh_trace(anchor, brick_type, rotation, group_color, brick_name, legendgroup)
        if mesh_d is not None:
            mesh_d.uid = f"brick-mesh-{example_idx}-{node_idx}"
            traces["bricks"]["dipoles"].append(mesh_d.to_plotly_json())
        if mesh_g is not None:
            mesh_g.uid = f"brick-mesh-group-{example_idx}-{node_idx}"
            traces["bricks"]["groups"].append(mesh_g.to_plotly_json())

        surface = _surface_trace_from_brick(
            sample=sample,
            anchors=anchors,
            types=types,
            rotations=rotations,
            features=shape_features,
            dipoles=dipole_vec,
            brick_index=node_idx,
        )
        surface.uid = f"brick-surface-{example_idx}-{node_idx}"
        traces["surfaces"]["dipoles"].append(surface.to_plotly_json())
        traces["surfaces"]["groups"].append(surface.to_plotly_json())

    return {
        "scene": _scene_spec(sample, projection=projection),
        "num_nodes": num_nodes,
        "num_designable": int(ligand_mask.sum()),
        "source_frame_id": source_frame_id,
        "split_id": split_id,
        "traces": traces,
    }


def _plot_examples_html(
    data: np.lib.npyio.NpzFile,
    indices: list[int],
    output_path: Path,
    designable_label: str,
    context_label: str,
    initial_projection: str,
    open_html: bool,
) -> None:
    if len(indices) == 0:
        raise ValueError("No example indices provided for HTML plotting.")

    total = int(data["pos"].shape[0])
    for idx in indices:
        if idx < 0 or idx >= total:
            raise IndexError(f"Example index {idx} is out of range for {total} examples.")

    states = []
    for example_idx in indices:
        payload = _example_traces(
            data=data,
            example_idx=example_idx,
            designable_label=designable_label,
            context_label=context_label,
            projection=initial_projection,
        )
        payload["example_idx"] = int(example_idx)
        states.append(payload)

    base_layout = {
        "title": {"text": "LEGO Diffusion Dataset Inspector", "x": 0.02, "xanchor": "left"},
        "template": "plotly_white",
        "margin": {"l": 0, "r": 0, "t": 58, "b": 0},
        "legend": {
            "orientation": "v",
            "x": 1.01,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "groupclick": "toggleitem",
            "itemdoubleclick": False,
            "uirevision": "legend",
        },
        "scene": states[0]["scene"],
        "uirevision": "inspect-diffusion",
    }
    fig = go.Figure(data=[], layout=base_layout)
    plot_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        div_id="inspect-plot",
        config={"responsive": True, "displaylogo": False},
    )

    options = "\n".join(
        (
            f'<option value="{idx}"{" selected" if idx == 0 else ""}>'
            f'Example {state["example_idx"]} · frame {state["source_frame_id"]} · split {state["split_id"]}'
            f"</option>"
        )
        for idx, state in enumerate(states)
    )
    state_json = json.dumps(states, cls=PlotlyJSONEncoder)
    layout_json = json.dumps(base_layout, cls=PlotlyJSONEncoder)
    designable_lower = str(designable_label).lower()
    context_lower = str(context_label).lower()
    projection_selected = {"orthographic": "", "perspective": ""}
    projection_selected[str(initial_projection)] = " selected"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LEGO Diffusion Inspector</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: #f6f2ea;
      color: #1e262d;
    }}
    .shell {{
      padding: 14px 16px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 280px));
      gap: 12px;
      align-items: end;
      margin-bottom: 10px;
    }}
    .control {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .control label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #5a6672;
    }}
    .control select {{
      font: inherit;
      border: 1px solid rgba(30, 38, 45, 0.18);
      border-radius: 10px;
      padding: 8px 10px;
      background: #fff;
      color: #1e262d;
    }}
    .meta {{
      margin: 0 0 10px 0;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(30, 38, 45, 0.08);
      font-size: 14px;
    }}
    #inspect-plot {{
      height: calc(100vh - 180px);
      min-height: 640px;
      border-radius: 16px;
      overflow: hidden;
      background: rgba(255,255,255,0.7);
      box-shadow: 0 18px 48px rgba(30, 38, 45, 0.10);
    }}
    @media (max-width: 920px) {{
      .controls {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      #inspect-plot {{
        min-height: 520px;
        height: calc(100vh - 240px);
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="controls">
      <div class="control">
        <label for="frame-select">Frame</label>
        <select id="frame-select">{options}</select>
      </div>
      <div class="control">
        <label for="geometry-select">Geometry</label>
        <select id="geometry-select">
          <option value="bricks" selected>Bricks</option>
          <option value="surfaces">Spherical-harmonic surfaces</option>
        </select>
      </div>
      <div class="control">
        <label for="coloring-select">Coloring</label>
        <select id="coloring-select">
          <option value="dipoles" selected>Dipoles</option>
          <option value="groups">{designable_label}/{context_label}</option>
        </select>
      </div>
      <div class="control">
        <label for="projection-select">Projection</label>
        <select id="projection-select">
          <option value="orthographic"{projection_selected["orthographic"]}>Orthographic</option>
          <option value="perspective"{projection_selected["perspective"]}>Perspective</option>
        </select>
      </div>
    </div>
    <div id="meta" class="meta"></div>
    {plot_html}
  </div>
  <script>
    const states = {state_json};
    const baseLayout = {layout_json};
    const plotEl = document.getElementById("inspect-plot");
    const frameSelect = document.getElementById("frame-select");
    const geometrySelect = document.getElementById("geometry-select");
    const coloringSelect = document.getElementById("coloring-select");
    const projectionSelect = document.getElementById("projection-select");
    const metaEl = document.getElementById("meta");

    function deepClone(value) {{
      return JSON.parse(JSON.stringify(value));
    }}

    function render() {{
      const state = states[Number(frameSelect.value)];
      const geometry = geometrySelect.value;
      const coloring = coloringSelect.value;
      const traces = deepClone(state.traces[geometry][coloring]).map((trace, idx) => {{
        trace.scene = "scene";
        trace.uid = (trace.uid || ("trace-" + idx)) + "-scene";
        return trace;
      }});

      const currentCamera = plotEl && plotEl._fullLayout && plotEl._fullLayout.scene
        ? plotEl._fullLayout.scene.camera
        : null;
      const layout = deepClone(baseLayout);
      layout.scene = deepClone(state.scene);
      if (currentCamera) {{
        layout.scene.camera = currentCamera;
      }}
      layout.scene.camera = layout.scene.camera || {{}};
      layout.scene.camera.projection = layout.scene.camera.projection || {{}};
      layout.scene.camera.projection.type = projectionSelect.value;
      layout.legend = layout.legend || {{}};
      layout.legend.uirevision = "legend";
      layout.uirevision = "inspect-diffusion";
      layout.scene.uirevision = "inspect-diffusion-scene";

      metaEl.innerHTML = `
        <strong>Example ${{state.example_idx}}</strong> ·
        frame ${{state.source_frame_id}} ·
        split ${{state.split_id}} ·
        ${{state.num_nodes}} nodes ·
        ${{state.num_designable}} {designable_lower} ·
        ${{state.num_nodes - state.num_designable}} {context_lower}
      `;
      Plotly.react(plotEl, traces, layout, {{responsive: true, displaylogo: false}});
    }}

    [frameSelect, geometrySelect, coloringSelect, projectionSelect].forEach((el) => el.addEventListener("change", render));
    render();
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    if bool(open_html):
        import webbrowser
        webbrowser.open(output_path.resolve().as_uri())
    print(f"\nSaved HTML plot to: {output_path}")


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        num_examples = int(data["pos"].shape[0])
        pos_mask = np.asarray(data["pos__mask__"], dtype=bool)
        num_nodes = np.asarray(data["num_nodes"], dtype=np.int64)
        ligand_size = np.asarray(data["ligand_size"], dtype=np.int64)
        num_edges = np.asarray(data["num_edges"], dtype=np.int64)
        dipole_strength = np.asarray(data["dipole_strength"], dtype=np.float32)
        dipole_mask = np.asarray(data["dipole_strength__mask__"], dtype=bool)

        avg_nodes = float(num_nodes.mean()) if num_examples > 0 else 0.0
        avg_ligand = float(ligand_size.mean()) if num_examples > 0 else 0.0
        avg_edges_per_node = float(np.mean(num_edges / np.maximum(num_nodes, 1))) if num_examples > 0 else 0.0

        histogram = Counter(
            int(float(value) > 1e-6)
            for strengths, mask in zip(dipole_strength, dipole_mask)
            for value in strengths[_valid_nodes(mask)].reshape(-1).tolist()
        )

        print("--- Dataset Stats ---")
        print(f"Examples: {num_examples}")
        print(f"Average nodes/example: {avg_nodes:.2f}")
        print(f"Average ligand size: {avg_ligand:.2f}")
        print(f"Average directed contacts/node: {avg_edges_per_node:.2f}")
        print("Dipole state histogram:")
        print(f"  neutral: {histogram.get(0, 0)}")
        print(f"  polar: {histogram.get(1, 0)}")

        plot_indices = sorted(set(_parse_indices_arg(args.indices)))
        if len(plot_indices) == 0:
            plot_indices = list(range(num_examples))

        output_html = args.plot_html if args.plot_html is not None else args.input.with_suffix(".html")
        _plot_examples_html(
            data=data,
            indices=plot_indices,
            output_path=output_html,
            designable_label=str(args.designable_label),
            context_label=str(args.context_label),
            initial_projection=str(args.initial_projection),
            open_html=bool(args.open_html),
        )


if __name__ == "__main__":
    main()
