from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from lego.lego_blocks import LEGO_LIBRARY, NEIGHBOR_DIRS, get_exposed_faces, rotated_offsets
except ModuleNotFoundError:
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from lego.lego_blocks import LEGO_LIBRARY, NEIGHBOR_DIRS, get_exposed_faces, rotated_offsets


def _as_int_rotation(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    return np.rint(rotation).astype(np.int32)


def _canonical_sort_rows(array: np.ndarray, decimals: int = 5) -> np.ndarray:
    array = np.asarray(array)
    if array.size == 0:
        return array.reshape(0, array.shape[-1] if array.ndim > 1 else 0)
    rounded = np.round(array.astype(np.float32), decimals=decimals)
    order = np.lexsort(tuple(rounded[:, axis] for axis in range(rounded.shape[1] - 1, -1, -1)))
    return array[order]


def compute_intrinsic_ports(brick_type: str, rotation: np.ndarray) -> np.ndarray:
    offsets = np.asarray(LEGO_LIBRARY[brick_type]["offsets"], dtype=np.int32)
    rotated = rotated_offsets(offsets, _as_int_rotation(rotation)).astype(np.float32)
    center = rotated.mean(axis=0)
    ports = get_exposed_faces(rotated) - center
    return _canonical_sort_rows(ports.astype(np.float32))


def build_brick_geometries(sample: Dict) -> List[Dict[str, np.ndarray]]:
    anchors = np.asarray(sample["brick_anchors"], dtype=np.float32)
    rotations = np.asarray(sample["brick_rotations"], dtype=np.float32)
    brick_types = np.asarray(sample["brick_types"])

    geometries: List[Dict[str, np.ndarray]] = []
    for brick_index, (anchor, rotation, brick_type) in enumerate(zip(anchors, rotations, brick_types)):
        brick_name = str(brick_type)
        offsets = np.asarray(LEGO_LIBRARY[brick_name]["offsets"], dtype=np.int32)
        local_offsets = rotated_offsets(offsets, _as_int_rotation(rotation)).astype(np.int32)
        local_center = local_offsets.astype(np.float32).mean(axis=0)
        world_center = anchor.astype(np.float32) + local_center
        world_voxels = local_offsets.astype(np.float32) + anchor.astype(np.float32)
        intrinsic_ports = compute_intrinsic_ports(brick_name, rotation)
        geometries.append(
            {
                "brick_index": np.asarray(brick_index, dtype=np.int64),
                "brick_type": np.asarray(brick_name),
                "anchor": anchor.astype(np.float32),
                "rotation": np.asarray(rotation, dtype=np.float32),
                "local_offsets": local_offsets.astype(np.int32),
                "local_center": local_center.astype(np.float32),
                "world_center": world_center.astype(np.float32),
                "world_voxels": world_voxels.astype(np.float32),
                "intrinsic_ports": intrinsic_ports.astype(np.float32),
            }
        )
    return geometries


def reconstruct_world_faces(geometries: Sequence[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    faces: List[Dict[str, np.ndarray]] = []
    for geometry in geometries:
        local_offsets = np.asarray(geometry["local_offsets"], dtype=np.int32)
        local_center = np.asarray(geometry["local_center"], dtype=np.float32)
        anchor = np.asarray(geometry["anchor"], dtype=np.float32)
        for voxel in local_offsets:
            for direction in NEIGHBOR_DIRS:
                neighbor = tuple((voxel + direction).tolist())
                if neighbor in {tuple(v.tolist()) for v in local_offsets}:
                    continue
                face_center_local = voxel.astype(np.float32) + 0.5 * direction.astype(np.float32)
                faces.append(
                    {
                        "brick_index": np.asarray(int(geometry["brick_index"]), dtype=np.int64),
                        "face_center_world": (anchor + face_center_local).astype(np.float32),
                        "face_normal_world": direction.astype(np.int32),
                        "face_center_local_centered": (face_center_local - local_center).astype(np.float32),
                    }
                )
    return faces


def build_world_voxel_owner_map(
    geometries: Sequence[Dict[str, np.ndarray]],
) -> Tuple[Dict[Tuple[int, int, int], int], Dict[int, Dict[Tuple[int, int, int], np.ndarray]]]:
    owner_map: Dict[Tuple[int, int, int], int] = {}
    local_port_map: Dict[int, Dict[Tuple[int, int, int], np.ndarray]] = {}

    for geometry in geometries:
        brick_index = int(geometry["brick_index"])
        anchor = np.asarray(geometry["anchor"], dtype=np.float32)
        local_center = np.asarray(geometry["local_center"], dtype=np.float32)
        local_ports: Dict[Tuple[int, int, int], np.ndarray] = {}
        for voxel in np.asarray(geometry["local_offsets"], dtype=np.int32):
            world_voxel = tuple(np.rint(voxel.astype(np.float32) + anchor).astype(np.int32).tolist())
            if world_voxel in owner_map:
                raise ValueError(
                    f"Overlapping bricks detected at voxel {world_voxel}: "
                    f"{owner_map[world_voxel]} and {brick_index}"
                )
            owner_map[world_voxel] = brick_index
            local_ports[tuple(voxel.tolist())] = voxel.astype(np.float32) - local_center
        local_port_map[brick_index] = local_ports
    return owner_map, local_port_map


def detect_brick_contacts(
    geometries: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    owner_map, _ = build_world_voxel_owner_map(geometries)
    used_ports_by_brick: List[List[np.ndarray]] = [[] for _ in geometries]
    unique_pairs: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    face_contact_pairs: List[Tuple[int, int]] = []
    face_contact_dirs: List[np.ndarray] = []

    centers = [np.asarray(geometry["world_center"], dtype=np.float32) for geometry in geometries]

    for world_voxel, brick_index in owner_map.items():
        voxel = np.asarray(world_voxel, dtype=np.int32)
        for direction in NEIGHBOR_DIRS:
            neighbor_key = tuple((voxel + direction).tolist())
            neighbor_brick = owner_map.get(neighbor_key)
            if neighbor_brick is None or neighbor_brick == brick_index:
                continue

            face_center_world = voxel.astype(np.float32) + 0.5 * direction.astype(np.float32)
            used_ports_by_brick[brick_index].append(face_center_world - centers[brick_index])

            if brick_index < neighbor_brick:
                unique_pairs[(brick_index, neighbor_brick)].append(direction.astype(np.int32))
                face_contact_pairs.append((int(brick_index), int(neighbor_brick)))
                face_contact_dirs.append(direction.astype(np.int32))

    used_ports = []
    used_port_counts = []
    for ports in used_ports_by_brick:
        if len(ports) == 0:
            used_ports.append(np.zeros((0, 3), dtype=np.float32))
            used_port_counts.append(0)
            continue
        stacked = _canonical_sort_rows(np.asarray(ports, dtype=np.float32))
        used_ports.append(stacked.astype(np.float32))
        used_port_counts.append(int(stacked.shape[0]))

    contact_pairs = np.asarray(sorted(unique_pairs.keys()), dtype=np.int64)
    if contact_pairs.size == 0:
        contact_pairs = contact_pairs.reshape(0, 2)
        contact_face_dirs = np.zeros((0, 3), dtype=np.int32)
    else:
        contact_face_dirs = np.asarray(
            [
                _canonical_sort_rows(np.asarray(unique_pairs[pair], dtype=np.int32), decimals=0)[0]
                for pair in sorted(unique_pairs.keys())
            ],
            dtype=np.int32,
        )

    if len(face_contact_pairs) == 0:
        all_face_contact_pairs = np.zeros((0, 2), dtype=np.int64)
        all_face_contact_dirs = np.zeros((0, 3), dtype=np.int32)
    else:
        face_records = sorted(
            zip(face_contact_pairs, face_contact_dirs),
            key=lambda item: (item[0][0], item[0][1], int(item[1][0]), int(item[1][1]), int(item[1][2])),
        )
        all_face_contact_pairs = np.asarray([pair for pair, _ in face_records], dtype=np.int64)
        all_face_contact_dirs = np.asarray([direction for _, direction in face_records], dtype=np.int32)

    edge_index, edge_types = build_contact_graph(len(geometries), contact_pairs)
    component_id = connected_components(len(geometries), contact_pairs)

    return {
        "used_ports": np.asarray(used_ports, dtype=object),
        "used_port_counts": np.asarray(used_port_counts, dtype=np.int32),
        "contact_pairs": contact_pairs,
        "contact_face_dirs": contact_face_dirs,
        "all_face_contact_pairs": all_face_contact_pairs,
        "all_face_contact_dirs": all_face_contact_dirs,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "component_id": component_id,
    }


def detect_split_interface_ports(
    geometries: Sequence[Dict[str, np.ndarray]],
    ligand_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    owner_map, _ = build_world_voxel_owner_map(geometries)
    ligand_mask = np.asarray(ligand_mask, dtype=bool).reshape(-1)
    if ligand_mask.shape[0] != len(geometries):
        raise ValueError(
            f"Expected ligand mask of length {len(geometries)}, got {ligand_mask.shape[0]}."
        )

    interface_ports_by_brick: List[List[np.ndarray]] = [[] for _ in geometries]
    interface_pairs = set()
    centers = [np.asarray(geometry["world_center"], dtype=np.float32) for geometry in geometries]

    for world_voxel, brick_index in owner_map.items():
        voxel = np.asarray(world_voxel, dtype=np.int32)
        brick_is_ligand = bool(ligand_mask[int(brick_index)])
        for direction in NEIGHBOR_DIRS:
            neighbor_key = tuple((voxel + direction).tolist())
            neighbor_brick = owner_map.get(neighbor_key)
            if neighbor_brick is None or neighbor_brick == brick_index:
                continue
            if bool(ligand_mask[int(neighbor_brick)]) == brick_is_ligand:
                continue

            face_center_world = voxel.astype(np.float32) + 0.5 * direction.astype(np.float32)
            interface_ports_by_brick[int(brick_index)].append(face_center_world - centers[int(brick_index)])
            interface_pairs.add(tuple(sorted((int(brick_index), int(neighbor_brick)))))

    interface_ports = []
    interface_port_counts = []
    for ports in interface_ports_by_brick:
        if len(ports) == 0:
            interface_ports.append(np.zeros((0, 3), dtype=np.float32))
            interface_port_counts.append(0)
            continue
        stacked = _canonical_sort_rows(np.asarray(ports, dtype=np.float32))
        interface_ports.append(stacked.astype(np.float32))
        interface_port_counts.append(int(stacked.shape[0]))

    if len(interface_pairs) == 0:
        interface_contact_pairs = np.zeros((0, 2), dtype=np.int64)
    else:
        interface_contact_pairs = np.asarray(sorted(interface_pairs), dtype=np.int64)

    return {
        "interface_ports": np.asarray(interface_ports, dtype=object),
        "interface_port_counts": np.asarray(interface_port_counts, dtype=np.int32),
        "interface_contact_pairs": interface_contact_pairs,
    }


def build_contact_graph(num_nodes: int, contact_pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if num_nodes <= 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int32)
    if contact_pairs.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int32)

    directed_edges = []
    for src, dst in np.asarray(contact_pairs, dtype=np.int64):
        if src == dst:
            raise ValueError(f"Self-contact detected for node {src}.")
        directed_edges.append((int(src), int(dst)))
        directed_edges.append((int(dst), int(src)))

    edge_index = np.asarray(directed_edges, dtype=np.int64).T
    edge_types = np.ones((edge_index.shape[1],), dtype=np.int32)
    return edge_index, edge_types


def connected_components(num_nodes: int, contact_pairs: np.ndarray) -> np.ndarray:
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for src, dst in np.asarray(contact_pairs, dtype=np.int64):
        adjacency[int(src)].append(int(dst))
        adjacency[int(dst)].append(int(src))

    component_id = np.full((num_nodes,), fill_value=-1, dtype=np.int32)
    current_component = 0
    for start in range(num_nodes):
        if component_id[start] != -1:
            continue
        stack = [start]
        component_id[start] = current_component
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if component_id[neighbor] != -1:
                    continue
                component_id[neighbor] = current_component
                stack.append(neighbor)
        current_component += 1
    return component_id
