from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


ROLE_NAMES: List[str] = [
    "CAP_START",
    "CAP_END",
    "STRAIGHT",
    "BEND_LEFT",
    "BEND_RIGHT",
    "PLANAR",
    "SHEET_EDGE",
    "JUNCTION_T",
    "JUNCTION_BRANCH_LEFT",
    "JUNCTION_BRANCH_RIGHT",
    "HELIX_PHASE_0",
    "HELIX_PHASE_1",
    "HELIX_PHASE_2",
    "HELIX_PHASE_3",
]
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLE_NAMES)}


def _branch_orders(branch_id: np.ndarray, seq_index_in_branch: np.ndarray) -> Dict[int, np.ndarray]:
    branch_id = np.asarray(branch_id, dtype=np.int32).reshape(-1)
    seq_index_in_branch = np.asarray(seq_index_in_branch, dtype=np.int32).reshape(-1)
    orders: Dict[int, np.ndarray] = {}
    for branch in sorted(np.unique(branch_id).tolist()):
        nodes = np.flatnonzero(branch_id == int(branch)).astype(np.int32)
        order = nodes[np.argsort(seq_index_in_branch[nodes], kind="stable")]
        orders[int(branch)] = order.astype(np.int32)
    return orders


def _junction_child_roots(parent_id: np.ndarray, node_index: int) -> np.ndarray:
    parent_id = np.asarray(parent_id, dtype=np.int32).reshape(-1)
    return np.flatnonzero(parent_id == int(node_index)).astype(np.int32)


def _signed_turn(parent_tangent: np.ndarray, child_tangent: np.ndarray, normal: np.ndarray) -> float:
    parent_tangent = np.asarray(parent_tangent, dtype=np.float32)
    child_tangent = np.asarray(child_tangent, dtype=np.float32)
    normal = np.asarray(normal, dtype=np.float32)
    cross = np.cross(parent_tangent, child_tangent).astype(np.float32)
    return float(np.dot(cross, normal))


def assign_roles(
    *,
    parent_id: np.ndarray,
    branch_id: np.ndarray,
    seq_index_in_branch: np.ndarray,
    branch_kind: np.ndarray,
    degree_topology: np.ndarray,
    descriptors: Dict[str, np.ndarray],
    tau_straight: float = 0.35,
    tau_planar: float = 0.34,
    junction_degree_threshold: int = 3,
    helix_phase_period: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    parent_id = np.asarray(parent_id, dtype=np.int32).reshape(-1)
    branch_id = np.asarray(branch_id, dtype=np.int32).reshape(-1)
    seq_index_in_branch = np.asarray(seq_index_in_branch, dtype=np.int32).reshape(-1)
    branch_kind = np.asarray(branch_kind).astype(str).reshape(-1)
    degree_topology = np.asarray(degree_topology, dtype=np.int32).reshape(-1)
    n = int(parent_id.shape[0])

    tangent = np.asarray(descriptors["tangent"], dtype=np.float32)
    curvature = np.asarray(descriptors["curvature_mag"], dtype=np.float32).reshape(-1)
    bend_axis = np.asarray(descriptors["bend_axis"], dtype=np.float32)
    planarity = np.asarray(descriptors["planarity_score"], dtype=np.float32).reshape(-1)
    boundary_flag = np.asarray(descriptors["boundary_flag"], dtype=bool).reshape(-1)
    branch_local_normal = np.asarray(descriptors["branch_local_normal"], dtype=np.float32)
    next_same_branch = np.asarray(descriptors["next_same_branch"], dtype=np.int32).reshape(-1)

    roles = np.full((n,), fill_value=ROLE_TO_ID["STRAIGHT"], dtype=np.int32)
    locked = np.zeros((n,), dtype=bool)
    junction_nodes = np.flatnonzero(degree_topology >= int(max(2, junction_degree_threshold))).astype(np.int32)

    # Junction anchors and immediate branch starts.
    for node in junction_nodes.tolist():
        roles[int(node)] = ROLE_TO_ID["JUNCTION_T"]
        locked[int(node)] = True
        child_roots = _junction_child_roots(parent_id=parent_id, node_index=int(node))
        if child_roots.size == 0:
            continue
        parent_t = tangent[int(node)]
        normal = branch_local_normal[int(node)]
        signed = []
        for child in child_roots.tolist():
            sign = _signed_turn(parent_t, tangent[int(child)], normal)
            signed.append((float(sign), int(child)))
        signed.sort(key=lambda item: item[0], reverse=True)
        if len(signed) > 0:
            _, child_left = signed[0]
            roles[int(child_left)] = ROLE_TO_ID["JUNCTION_BRANCH_LEFT"]
            locked[int(child_left)] = True
            child2 = next_same_branch[int(child_left)]
            if child2 >= 0:
                roles[int(child2)] = ROLE_TO_ID["JUNCTION_BRANCH_LEFT"]
                locked[int(child2)] = True
        if len(signed) > 1:
            _, child_right = signed[-1]
            roles[int(child_right)] = ROLE_TO_ID["JUNCTION_BRANCH_RIGHT"]
            locked[int(child_right)] = True
            child2 = next_same_branch[int(child_right)]
            if child2 >= 0:
                roles[int(child2)] = ROLE_TO_ID["JUNCTION_BRANCH_RIGHT"]
                locked[int(child2)] = True

    # First-pass geometry rules.
    for node in range(n):
        if locked[node]:
            continue
        if bool(boundary_flag[node]) and int(seq_index_in_branch[node]) == 0 and int(parent_id[node]) < 0:
            roles[node] = ROLE_TO_ID["CAP_START"]
            locked[node] = True
            continue
        if bool(boundary_flag[node]) and int(next_same_branch[node]) < 0:
            roles[node] = ROLE_TO_ID["CAP_END"]
            locked[node] = True
            continue

        if float(planarity[node]) >= float(tau_planar):
            near_branch_boundary = int(seq_index_in_branch[node]) <= 1 or int(next_same_branch[node]) < 0
            roles[node] = ROLE_TO_ID["SHEET_EDGE"] if near_branch_boundary else ROLE_TO_ID["PLANAR"]
            continue

        if str(branch_kind[node]).lower().startswith("helix"):
            phase = int(seq_index_in_branch[node] % int(max(1, helix_phase_period)))
            roles[node] = ROLE_TO_ID[f"HELIX_PHASE_{phase % 4}"]
            continue

        if float(curvature[node]) < float(tau_straight):
            roles[node] = ROLE_TO_ID["STRAIGHT"]
            continue

        signed_bend = float(np.dot(bend_axis[node], branch_local_normal[node]))
        roles[node] = ROLE_TO_ID["BEND_LEFT"] if signed_bend >= 0.0 else ROLE_TO_ID["BEND_RIGHT"]

    # Second pass deterministic regularization.
    orders = _branch_orders(branch_id=branch_id, seq_index_in_branch=seq_index_in_branch)
    protected = {
        ROLE_TO_ID["CAP_START"],
        ROLE_TO_ID["CAP_END"],
        ROLE_TO_ID["JUNCTION_T"],
        ROLE_TO_ID["JUNCTION_BRANCH_LEFT"],
        ROLE_TO_ID["JUNCTION_BRANCH_RIGHT"],
    }

    for branch, order in orders.items():
        if order.size < 3:
            continue
        for local_idx in range(1, int(order.size) - 1):
            center = int(order[local_idx])
            if roles[center] in protected:
                continue
            left = int(order[local_idx - 1])
            right = int(order[local_idx + 1])
            if roles[left] == roles[right] and roles[center] != roles[left]:
                roles[center] = roles[left]

        kind = str(branch_kind[int(order[0])]).lower()
        if kind.startswith("helix"):
            for node in order.tolist():
                node_i = int(node)
                if roles[node_i] in protected:
                    continue
                phase = int(seq_index_in_branch[node_i] % int(max(1, helix_phase_period)))
                roles[node_i] = ROLE_TO_ID[f"HELIX_PHASE_{phase % 4}"]

    # Enforce exactly one JUNCTION_T label per high-degree node.
    for node in junction_nodes.tolist():
        roles[int(node)] = ROLE_TO_ID["JUNCTION_T"]

    role_names = np.asarray([ROLE_NAMES[int(role_id)] for role_id in roles.tolist()], dtype=f"<U{max(len(x) for x in ROLE_NAMES)}")
    return roles.astype(np.int32), role_names

