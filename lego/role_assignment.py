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
    """Assign deterministic brick roles from scaffold labels and descriptors.

    This version is intentionally grammar-driven:

    * beta_sheet: explicit turn nodes are L-shapes; straight nodes alternate
      SHEET_EDGE/PLANAR, which map to 1x1/1x2. `chain` is a legacy alias.
    * beta_sheet-like motifs: every internal node is a T-shape, alternating
      orientation through JUNCTION_BRANCH_LEFT/JUNCTION_BRANCH_RIGHT.
      Curvature no longer converts beta-sheet turns into L-shapes.
    * helix: helix phase is assigned before generic planarity/curvature rules,
      so a thin continuous helix cannot be accidentally reclassified as planar.
    """
    parent_id = np.asarray(parent_id, dtype=np.int32).reshape(-1)
    branch_id = np.asarray(branch_id, dtype=np.int32).reshape(-1)
    seq_index_in_branch = np.asarray(seq_index_in_branch, dtype=np.int32).reshape(-1)
    branch_kind = np.asarray(branch_kind).astype(str).reshape(-1)
    degree_topology = np.asarray(degree_topology, dtype=np.int32).reshape(-1)
    n = int(parent_id.shape[0])

    curvature = np.asarray(descriptors["curvature_mag"], dtype=np.float32).reshape(-1)
    bend_axis = np.asarray(descriptors["bend_axis"], dtype=np.float32)
    planarity = np.asarray(descriptors["planarity_score"], dtype=np.float32).reshape(-1)
    boundary_flag = np.asarray(descriptors["boundary_flag"], dtype=bool).reshape(-1)
    branch_local_normal = np.asarray(descriptors["branch_local_normal"], dtype=np.float32)
    next_same_branch = np.asarray(descriptors["next_same_branch"], dtype=np.int32).reshape(-1)
    sheet_turn_mask = np.asarray(
        descriptors.get("sheet_turn_mask", np.zeros((n,), dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    sheet_segment_id = np.asarray(
        descriptors.get("sheet_segment_id", np.full((n,), fill_value=-1, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)

    roles = np.full((n,), fill_value=ROLE_TO_ID["STRAIGHT"], dtype=np.int32)

    for node in range(n):
        kind = str(branch_kind[node]).lower()
        seq = int(seq_index_in_branch[node])
        is_start = bool(boundary_flag[node]) and seq == 0 and int(parent_id[node]) < 0
        is_end = bool(boundary_flag[node]) and int(next_same_branch[node]) < 0

        if kind.startswith("helix"):
            # Keep endpoints compact, but make every internal helix node phase-driven.
            if is_start:
                roles[node] = ROLE_TO_ID["CAP_START"]
            elif is_end:
                roles[node] = ROLE_TO_ID["CAP_END"]
            else:
                phase = int(seq % int(max(1, helix_phase_period)))
                roles[node] = ROLE_TO_ID[f"HELIX_PHASE_{phase % 4}"]
            continue

        if kind.startswith("beta_sheet") or kind.startswith("chain"):
            if is_start:
                roles[node] = ROLE_TO_ID["CAP_START"]
            elif is_end:
                roles[node] = ROLE_TO_ID["CAP_END"]
            else:
                # Alternating T-up/T-down. shape_prototypes.py maps both roles to
                # T-shape and flips the local radial axis for the RIGHT role.
                t_index = max(0, seq - 1)
                roles[node] = (
                    ROLE_TO_ID["JUNCTION_BRANCH_LEFT"]
                    if (t_index % 2 == 0)
                    else ROLE_TO_ID["JUNCTION_BRANCH_RIGHT"]
                )
            continue

        if kind.startswith("beta_sheet") or kind.startswith("sheet"):
            if bool(sheet_turn_mask[node]):
                signed_bend = float(np.dot(bend_axis[node], branch_local_normal[node]))
                roles[node] = ROLE_TO_ID["BEND_LEFT"] if signed_bend >= 0.0 else ROLE_TO_ID["BEND_RIGHT"]
            else:
                # Temporary assignment; the final segment-specific pass below
                # enforces 1x1/1x2 alternation within each straight run.
                roles[node] = ROLE_TO_ID["SHEET_EDGE"]
            continue

        # Fallback for older/mixed scaffolds.
        if is_start:
            roles[node] = ROLE_TO_ID["CAP_START"]
            continue
        if is_end:
            roles[node] = ROLE_TO_ID["CAP_END"]
            continue
        if float(planarity[node]) >= float(tau_planar):
            near_branch_boundary = seq <= 1 or int(next_same_branch[node]) < 0
            roles[node] = ROLE_TO_ID["SHEET_EDGE"] if near_branch_boundary else ROLE_TO_ID["PLANAR"]
            continue
        if float(curvature[node]) < float(tau_straight):
            roles[node] = ROLE_TO_ID["STRAIGHT"]
            continue
        signed_bend = float(np.dot(bend_axis[node], branch_local_normal[node]))
        roles[node] = ROLE_TO_ID["BEND_LEFT"] if signed_bend >= 0.0 else ROLE_TO_ID["BEND_RIGHT"]

    # Explicit beta-sheet alternation: 1x1/1x2/1x1/1x2 along each straight segment.
    orders = _branch_orders(branch_id=branch_id, seq_index_in_branch=seq_index_in_branch)
    for _, order in orders.items():
        if order.size < 2:
            continue
        kind = str(branch_kind[int(order[0])]).lower()
        if not (kind.startswith("beta_sheet") or kind.startswith("sheet")):
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

    # Compatibility guard for legacy branching topologies.
    junction_nodes = np.flatnonzero(degree_topology >= int(max(2, junction_degree_threshold))).astype(np.int32)
    for node in junction_nodes.tolist():
        roles[int(node)] = ROLE_TO_ID["JUNCTION_T"]

    role_names = np.asarray(
        [ROLE_NAMES[int(role_id)] for role_id in roles.tolist()],
        dtype=f"<U{max(len(x) for x in ROLE_NAMES)}",
    )
    return roles.astype(np.int32), role_names
