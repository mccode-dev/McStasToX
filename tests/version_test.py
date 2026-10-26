# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)


import h5py
import numpy as np
import pytest

import mcstastox
from mcstastox.ReadNeXus import _MCSTAS_VERSION_SETTINGS


def _write(path, program: str) -> None:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry1")
        for key in ("data", "simulation/Param", "instrument/components"):
            entry.require_group(key)
        # fixed-length string attribute, as McStas writes it
        entry["simulation"].attrs["program"] = np.bytes_(program)


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("McStas 3.5.24 - Jun. 01, 2026", (3, 5, 24)),
        (" 3.99.99, git", (3, 99, 99)),
        ("McStas 3.10.0 (mccode-antlr 0.36.0)", (3, 10, 0)),  # McCode version first
    ],
)
def test_version_from_program(tmp_path, program, expected) -> None:
    _write(tmp_path / "mccode.h5", program)
    with mcstastox.Read(tmp_path) as data:
        assert data.file_object.mcstas_version == expected


def test_no_version_assumes_newest_layout(tmp_path) -> None:
    _write(tmp_path / "mccode.h5", "McStas")  # e.g. mccode-antlr <= 0.36
    with pytest.warns(UserWarning, match="No McStas version"):
        with mcstastox.Read(tmp_path) as data:
            assert data.file_object.mcstas_version == max(_MCSTAS_VERSION_SETTINGS)


def test_read_passes_mcstas_version(tmp_path) -> None:
    _write(tmp_path / "mccode.h5", "McStas")
    with mcstastox.Read(tmp_path, mcstas_version=(3, 6, 0)) as data:
        assert data.file_object.mcstas_version == (3, 6, 0)
