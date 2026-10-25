# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)


import h5py
import pytest

from mcstastox.ReadNeXus import _validate_file


def _make_entry(f: h5py.File, name: str) -> None:
    group = f.create_group(name)
    for key in ("data", "simulation/Param", "instrument/components"):
        group.require_group(key)


def test_validate_file_entry(tmp_path) -> None:
    # a McStas NeXus scan writes one NXentry per step
    with h5py.File(tmp_path / "scan.h5", "w") as f:
        _make_entry(f, "entry1")
        _make_entry(f, "entry2")
        _validate_file(f)
        _validate_file(f, "entry2")
        with pytest.raises(ValueError, match="entry3"):
            _validate_file(f, "entry3")
