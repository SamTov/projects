"""Luzar–Chandler hydrogen-bond graph for water configurations.

Reference: Luzar & Chandler, Phys. Rev. Lett. 76, 928 (1996).

An O_d → O_a hydrogen bond exists when
    r(O_d, O_a) < r_oo_cut                (Å)
and
    angle between (O_d → H) and (O_d → O_a) vectors < angle_cut    (degrees)
for either of O_d's two covalently-bonded hydrogens. The result is
returned as a directed adjacency matrix; the undirected H-bond degree
per oxygen is in-degree + out-degree.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "minimum_image",
    "hbond_adjacency_luzar_chandler",
    "hbond_degree",
    "time_averaged_hbond_degree",
]


def minimum_image(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply minimum-image convention along the last axis for a cubic box."""
    return dr - box * np.round(dr / box)


def hbond_adjacency_luzar_chandler(
    o_positions: np.ndarray,         # (N_O, 3)
    h_positions: np.ndarray,         # (N_H, 3)
    o_to_h_pairs: np.ndarray,        # (N_O, 2) int — indices into h_positions
    box: np.ndarray,                 # (3,) lengths
    r_oo_cut: float = 3.5,
    angle_cut_deg: float = 30.0,
) -> np.ndarray:
    """Return a (N_O, N_O) directed 0/1 adjacency matrix.

    `adj[d, a] == 1` iff oxygen `d` donates a hydrogen bond to oxygen `a`.
    Self-bonds (d == a) are excluded.
    """
    n_o = o_positions.shape[0]
    if n_o == 0:
        return np.zeros((0, 0), dtype=np.int8)
    box = np.asarray(box, dtype=np.float64)
    cos_cut = float(np.cos(np.deg2rad(angle_cut_deg)))

    # Pairwise O-O distances under minimum image.
    dr = o_positions[:, None, :] - o_positions[None, :, :]
    dr = minimum_image(dr, box)
    d_oo = np.linalg.norm(dr, axis=-1)

    adj = np.zeros((n_o, n_o), dtype=np.int8)
    np.fill_diagonal(d_oo, np.inf)  # exclude self

    # Candidate donor-acceptor pairs.
    candidates = np.argwhere(d_oo < r_oo_cut)
    for d_idx, a_idx in candidates:
        # Vector from donor O to acceptor O (minimum image).
        v_oa = minimum_image(o_positions[a_idx] - o_positions[d_idx], box)
        norm_oa = float(np.linalg.norm(v_oa))
        if norm_oa <= 0:
            continue
        # Try both H atoms covalently bonded to donor.
        for h_local in o_to_h_pairs[d_idx]:
            v_oh = minimum_image(
                h_positions[int(h_local)] - o_positions[d_idx], box
            )
            norm_oh = float(np.linalg.norm(v_oh))
            if norm_oh <= 0:
                continue
            cos_a = float(np.dot(v_oh, v_oa) / (norm_oh * norm_oa))
            if cos_a >= cos_cut:  # angle <= angle_cut
                adj[d_idx, a_idx] = 1
                break  # one qualifying H is enough
    return adj


def hbond_degree(adj: np.ndarray) -> np.ndarray:
    """Total H-bond degree per oxygen: out-degree (donor) + in-degree (acceptor)."""
    adj = np.asarray(adj)
    return (adj.sum(axis=1) + adj.sum(axis=0)).astype(np.float64)


def time_averaged_hbond_degree(
    o_positions_frames: np.ndarray,    # (n_frames, N_O, 3)
    h_positions_frames: np.ndarray,    # (n_frames, N_H, 3)
    o_to_h_pairs: np.ndarray,
    box: np.ndarray,
    r_oo_cut: float = 3.5,
    angle_cut_deg: float = 30.0,
) -> np.ndarray:
    """Average per-oxygen H-bond degree across a sequence of frames.

    Returns a (N_O,) array of mean degrees.
    """
    n_frames = o_positions_frames.shape[0]
    if n_frames == 0:
        raise ValueError("need at least one frame")
    n_o = o_positions_frames.shape[1]
    accum = np.zeros(n_o, dtype=np.float64)
    for f in range(n_frames):
        adj = hbond_adjacency_luzar_chandler(
            o_positions_frames[f], h_positions_frames[f],
            o_to_h_pairs, box, r_oo_cut=r_oo_cut, angle_cut_deg=angle_cut_deg,
        )
        accum += hbond_degree(adj)
    return accum / float(n_frames)
