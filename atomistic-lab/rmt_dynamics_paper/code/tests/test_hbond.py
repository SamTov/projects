"""Tests for rmt_dynamics.hbond (Luzar–Chandler graph)."""
from __future__ import annotations

import numpy as np

from rmt_dynamics import (
    hbond_adjacency_luzar_chandler,
    hbond_degree,
    time_averaged_hbond_degree,
)


def _make_dimer():
    """Two water molecules positioned as a classic H-bond donor/acceptor pair.

    O0 at origin. Its H atoms point roughly along +y; one of them points
    almost exactly at O1, which sits 2.8 Å away along +y. The angle ∠H-O0-O1
    should be small (~5°), well below the 30° cutoff. r(O0-O1) = 2.8 < 3.5.
    """
    box = np.array([20.0, 20.0, 20.0])
    o_pos = np.array([
        [10.0, 10.0, 10.0],   # O0
        [10.0, 12.8, 10.0],   # O1 — 2.8 Å along +y
    ])
    # H atoms for O0: one pointing toward O1 (along +y), one pointing -y so it can't H-bond.
    h_pos = np.array([
        [10.0, 10.95, 10.0],   # H0a: 0.95 Å along +y (donates)
        [10.0,  9.05, 10.0],   # H0b: 0.95 Å along -y
        [10.7, 13.5, 10.0],    # H1a (O1's first H)
        [9.3,  13.5, 10.0],    # H1b
    ])
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    return o_pos, h_pos, pairs, box


def test_hbond_dimer_donor_to_acceptor():
    o_pos, h_pos, pairs, box = _make_dimer()
    adj = hbond_adjacency_luzar_chandler(o_pos, h_pos, pairs, box)
    # O0 donates to O1 (via H0a along +y).
    assert adj[0, 1] == 1
    # O1's H atoms point away from O0 — no reverse donation.
    assert adj[1, 0] == 0
    # No self-bonds.
    assert adj[0, 0] == 0
    assert adj[1, 1] == 0


def test_hbond_degree_counts_both_directions():
    o_pos, h_pos, pairs, box = _make_dimer()
    adj = hbond_adjacency_luzar_chandler(o_pos, h_pos, pairs, box)
    deg = hbond_degree(adj)
    # O0 = 1 (donor out), O1 = 1 (acceptor in).
    assert deg[0] == 1.0
    assert deg[1] == 1.0


def test_hbond_distance_cutoff_excludes_far_pairs():
    """Move O1 to 5 Å (> 3.5) along +y; no H-bond should remain."""
    o_pos, h_pos, pairs, box = _make_dimer()
    o_pos[1, 1] = 15.0
    adj = hbond_adjacency_luzar_chandler(o_pos, h_pos, pairs, box)
    assert adj.sum() == 0


def test_hbond_angle_cutoff_excludes_misaligned_donors():
    """Rotate O0's H atoms so neither points anywhere near O1."""
    o_pos, h_pos, pairs, box = _make_dimer()
    # H0a now points along +x rather than +y.
    h_pos[0] = [10.95, 10.0, 10.0]
    h_pos[1] = [9.05, 10.0, 10.0]
    adj = hbond_adjacency_luzar_chandler(o_pos, h_pos, pairs, box)
    assert adj[0, 1] == 0


def test_time_averaged_hbond_degree_constant_over_static_frames():
    o_pos, h_pos, pairs, box = _make_dimer()
    n_frames = 4
    o_frames = np.broadcast_to(o_pos, (n_frames, *o_pos.shape)).copy()
    h_frames = np.broadcast_to(h_pos, (n_frames, *h_pos.shape)).copy()
    deg = time_averaged_hbond_degree(o_frames, h_frames, pairs, box)
    # Same as single-frame degree because nothing moves.
    assert np.allclose(deg, [1.0, 1.0])
