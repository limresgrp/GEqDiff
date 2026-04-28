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


def _select_spaced(order: np.ndarray, candidates: List[int], min_gap: int) -> List[int]:
    if len(candidates) == 0:
        return []
    rank = {int(node): idx for idx, node in enumerate(order.tolist())}
    selected: List[int] = []
    for node in candidates:
        node_i = int(node)
        node_rank = int(rank.get(node_i, -10**9))
        if len(selected) == 0:
            selected.append(node_i)
            continue
        if all(abs(node_rank - int(rank[int(prev)])) >= int(min_gap) for prev in selected):
            selected.append(node_i)
    return selected


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
    t_shape_min_gap: int = 3,
    t_shape_curvature_threshold: float = 0.60,
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
    sheet_turn_mask = np.asarray(descriptors.get("sheet_turn_mask", np.zeros((n,), dtype=bool)), dtype=bool).reshape(-1)
    sheet_segment_id = np.asarray(descriptors.get("sheet_segment_id", np.full((n,), fill_value=-1, dtype=np.int32)), dtype=np.int32).reshape(-1)

    roles = np.full((n,), fill_value=ROLE_TO_ID["STRAIGHT"], dtype=np.int32)
    locked = np.zeros((n,), dtype=bool)

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

        kind = str(branch_kind[node]).lower()
        if kind.startswith("sheet"):
            # Sheet program: alternating 1x1/1x2 on each straight segment.
            # Explicit turn nodes from the generator always become L-shapes.
            if bool(sheet_turn_mask[node]) or float(curvature[node]) >= float(tau_straight):
                signed_bend = float(np.dot(bend_axis[node], branch_local_normal[node]))
                roles[node] = ROLE_TO_ID["BEND_LEFT"] if signed_bend >= 0.0 else ROLE_TO_ID["BEND_RIGHT"]
            else:
                roles[node] = ROLE_TO_ID["STRAIGHT"]
            continue

        if kind.startswith("chain"):
            # Chain program: alternating T-shapes with 1x1 in-between.
            # Odd indices host T-shapes, alternating up/down orientation labels.
            if float(curvature[node]) >= float(tau_straight):
                signed_bend = float(np.dot(bend_axis[node], branch_local_normal[node]))
                roles[node] = ROLE_TO_ID["BEND_LEFT"] if signed_bend >= 0.0 else ROLE_TO_ID["BEND_RIGHT"]
            else:
                seq = int(seq_index_in_branch[node])
                if seq % 2 == 1:
                    t_index = (seq - 1) // 2
                    roles[node] = ROLE_TO_ID["JUNCTION_BRANCH_LEFT"] if (t_index % 2 == 0) else ROLE_TO_ID["JUNCTION_BRANCH_RIGHT"]
                else:
                    roles[node] = ROLE_TO_ID["STRAIGHT"]
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
    }

    for branch, order in orders.items():
        if order.size < 3:
            continue
        kind = str(branch_kind[int(order[0])]).lower()
        if kind.startswith("chain") or kind.startswith("sheet"):
            continue
        for local_idx in range(1, int(order.size) - 1):
            center = int(order[local_idx])
            if roles[center] in protected:
                continue
            left = int(order[local_idx - 1])
            right = int(order[local_idx + 1])
            if roles[left] == roles[right] and roles[center] != roles[left]:
                roles[center] = roles[left]

        if kind.startswith("helix"):
            for node in order.tolist():
                node_i = int(node)
                if roles[node_i] in protected:
                    continue
                phase = int(seq_index_in_branch[node_i] % int(max(1, helix_phase_period)))
                roles[node_i] = ROLE_TO_ID[f"HELIX_PHASE_{phase % 4}"]

    # Promote a sparse set of turn-sensitive sites to T-shape role.
    # The label name is preserved for backward compatibility (`JUNCTION_T`),
    # but this role is now used for helix/turn local motifs rather than graph branching.
    min_gap = int(max(2, t_shape_min_gap))
    curvature_thr = float(max(0.0, t_shape_curvature_threshold))
    for _, order in orders.items():
        if order.size < 3:
            continue
        kind = str(branch_kind[int(order[0])]).lower()
        # Keep helix branches strictly phase-driven so alpha-helix motifs are
        # fully periodic and predictable from sequence index.
        if kind.startswith("helix") or kind.startswith("chain") or kind.startswith("sheet"):
            continue
        ranked: List[Tuple[float, int]] = []
        for node in order.tolist():
            node_i = int(node)
            if bool(boundary_flag[node_i]):
                continue
            if roles[node_i] in protected:
                continue
            if roles[node_i] not in {ROLE_TO_ID["BEND_LEFT"], ROLE_TO_ID["BEND_RIGHT"]}:
                continue
            if float(curvature[node_i]) < curvature_thr:
                continue
            score = float(curvature[node_i] + 0.25 * abs(float(np.dot(bend_axis[node_i], branch_local_normal[node_i]))))
            ranked.append((score, node_i))

        if len(ranked) == 0:
            continue
        ranked.sort(key=lambda item: item[0], reverse=True)
        ordered_candidates = [int(node) for _, node in ranked]
        selected = _select_spaced(order=order, candidates=ordered_candidates, min_gap=min_gap)
        # Keep density bounded for readability and to avoid overusing T-shapes.
        max_t = int(max(1, order.size // 6))
        for node_i in selected[:max_t]:
            roles[int(node_i)] = ROLE_TO_ID["JUNCTION_T"]

    # Compatibility guard: if a dataset still contains high-degree nodes,
    # force those nodes to the same historical role.
    junction_nodes = np.flatnonzero(degree_topology >= int(max(2, junction_degree_threshold))).astype(np.int32)
    for node in junction_nodes.tolist():
        roles[int(node)] = ROLE_TO_ID["JUNCTION_T"]

    # Final sheet-specific alternation pass. Straight nodes alternate 1x1/1x2
    # within each segment, while explicit turn nodes remain L-shapes.
    for branch, order in orders.items():
        if order.size < 2:
            continue
        kind = str(branch_kind[int(order[0])]).lower()
        if not kind.startswith("sheet"):
            continue
        segment_ids = sheet_segment_id[order]
        unique_segments = [int(seg) for seg in np.unique(segment_ids).tolist() if int(seg) >= 0]
        for seg in unique_segments:
            segment_nodes = order[segment_ids == int(seg)]
            if segment_nodes.size == 0:
                continue
            segment_nodes = segment_nodes[np.argsort(seq_index_in_branch[segment_nodes], kind="stable")]
            parity = 0
            for node_i in segment_nodes.tolist():
                idx = int(node_i)
                if bool(sheet_turn_mask[idx]):
                    signed_bend = float(np.dot(bend_axis[idx], branch_local_normal[idx]))
                    roles[idx] = ROLE_TO_ID["BEND_LEFT"] if signed_bend >= 0.0 else ROLE_TO_ID["BEND_RIGHT"]
                    continue
                roles[idx] = ROLE_TO_ID["SHEET_EDGE"] if (parity % 2 == 0) else ROLE_TO_ID["PLANAR"]
                parity += 1

    role_names = np.asarray([ROLE_NAMES[int(role_id)] for role_id in roles.tolist()], dtype=f"<U{max(len(x) for x in ROLE_NAMES)}")
    return roles.astype(np.int32), role_names
