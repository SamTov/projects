"""Tests for rmt_dynamics.io (C save/load roundtrip only; MDAnalysis-free)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rmt_dynamics import load_C, save_C


def test_save_load_roundtrip(tmp_path):
    C = np.array([[2.0, 1.0], [1.0, 3.0]])
    metadata = {"seed": 7, "experiment": "smoke"}
    path = tmp_path / "C.npz"
    save_C(path, C, metadata)

    C_loaded, md_loaded = load_C(path)
    assert np.array_equal(C, C_loaded)
    assert C_loaded.dtype == np.float64
    assert md_loaded["seed"] == 7
    assert md_loaded["experiment"] == "smoke"
    assert md_loaded["shape"] == [2, 2]


def test_save_rejects_non_npz(tmp_path):
    with pytest.raises(ValueError):
        save_C(tmp_path / "C.bin", np.eye(3), {})


def test_save_rejects_non_square(tmp_path):
    with pytest.raises(ValueError):
        save_C(tmp_path / "C.npz", np.zeros((3, 4)), {})


def test_load_missing_sidecar_returns_empty_metadata(tmp_path):
    path = tmp_path / "C.npz"
    C = np.eye(5)
    np.savez_compressed(path, C=C)  # no sidecar
    C_loaded, md = load_C(path)
    assert np.array_equal(C, C_loaded)
    assert md == {}


def test_save_produces_valid_sidecar_json(tmp_path):
    path = tmp_path / "C.npz"
    save_C(path, np.eye(4), {"key": "value"})
    with (path.with_suffix(".json")).open() as fh:
        parsed = json.load(fh)
    assert parsed["key"] == "value"
