# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)

import h5py
import numpy as np
import pytest

from mcstastox import Read


def _create_scan_file(path):
    with h5py.File(path, "w") as file:
        for entry_number, component_name in (
            (1, "first"),
            (2, "second"),
            (10, "tenth"),
        ):
            entry = file.create_group(f"entry{entry_number}")
            entry.create_group("data")

            simulation = entry.create_group("simulation")
            parameters = simulation.create_group("Param")
            parameters.create_dataset(
                "scan_value", data=np.array([np.bytes_(str(entry_number))])
            )
            parameters.create_dataset(
                "filename", data=np.array([np.bytes_(f"{component_name}.laz")])
            )
            simulation.attrs["program"] = np.bytes_(" 3.8.6, git")

            components = entry.create_group("instrument/components")
            components.create_group(f"0000_{component_name}")


def test_default_entry_and_entry_count(tmp_path):
    _create_scan_file(tmp_path / "mccode.h5")

    with Read(tmp_path) as loaded:
        assert loaded.get_components() == ["first"]
        assert loaded.get_number_of_entries() == 3
        assert loaded.get_instrument_parameters() == {
            "scan_value": 1.0,
            "filename": "first.laz",
        }


def test_load_requested_entry(tmp_path):
    _create_scan_file(tmp_path / "mccode.h5")

    with Read(tmp_path, entry_number=2) as loaded:
        assert loaded.get_components() == ["second"]
        assert loaded.get_instrument_parameters()["scan_value"] == 2.0


def test_entry_number_must_exist(tmp_path):
    _create_scan_file(tmp_path / "mccode.h5")

    with pytest.raises(ValueError, match="lacks 'entry3'"):
        Read(tmp_path, entry_number=3)
