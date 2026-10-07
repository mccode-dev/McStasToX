# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)
"""Tests of the Scipp exporters and additional event variable support.

The exporter logic is tested against a mocked Data instance, so no real
McStas NeXus file is required.
"""

import numpy as np
import pytest
import scipp as sc

import mcstastox
from mcstastox.LoadFile import Data, Transfer

N_EVENTS = 3


def _make_data():
    """Create a Data instance with mocked data access, no file required."""
    data = Data.__new__(Data)
    data.component_pixel_order = ["Square_1"]
    data.pixel_range = {"Square_1": [0, N_EVENTS - 1]}
    state = {"requested_variables": None}

    def get_event_data(variables, component_name=None, filter_zeros=True):
        state["requested_variables"] = list(variables)
        return {
            "p": np.array([1.0, 2.0, 3.0]),
            "t": np.array([10.0, 20.0, 30.0]),
            "id": np.array([0, 1, 2]),
            "L": np.array([1.5, 1.7, 1.9]),
            "x": np.array([-0.5, 0.0, 0.5]),
        }

    data.get_event_data = get_event_data
    data.get_id_to_global_coordinates = lambda component_name=None: np.array(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    data.get_global_component_coordinates = lambda component_name: np.array(
        [0.0, 0.0, 0.0]
    )
    return data, state


def _flatten_binned(array):
    """Flatten a binned scipp variable to a single numpy array."""
    return np.concatenate([np.asarray(value.values) for value in array.values])


def test_transfer_construction():
    transfer = Transfer(mcstas_variable="lambda", scipp_coord="energy", unit="eV")
    assert transfer.mcstas_variable == "lambda"
    assert transfer.scipp_coord == "energy"
    assert transfer.unit == "eV"


def test_transfer_defaults_to_no_unit():
    transfer = Transfer(mcstas_variable="lambda", scipp_coord="energy")
    assert transfer.unit is None


def test_transfer_exported_from_package_top_level():
    assert mcstastox.Transfer is Transfer


def test_show_components_with_geometry_prints_once(capsys):
    data = Data.__new__(Data)
    data.get_components_with_geometry = lambda: ["Square_1", "Banana_1"]

    data.show_components_with_geometry()

    assert capsys.readouterr().out == (
        "All components with geometry information in file:\nSquare_1\nBanana_1\n"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mcstas_variable": 5, "scipp_coord": "L", "unit": "angstrom"},
        {"mcstas_variable": "L", "scipp_coord": None, "unit": "angstrom"},
        {"mcstas_variable": "L", "scipp_coord": "energy", "unit": 3.0},
    ],
)
def test_transfer_rejects_non_string_fields(kwargs):
    with pytest.raises(TypeError):
        Transfer(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mcstas_variable": "", "scipp_coord": "L", "unit": "angstrom"},
        {"mcstas_variable": "L", "scipp_coord": "", "unit": "angstrom"},
        {"mcstas_variable": "L", "scipp_coord": "energy", "unit": ""},
    ],
)
def test_transfer_rejects_empty_fields(kwargs):
    with pytest.raises(ValueError, match="must not be empty"):
        Transfer(**kwargs)


def test_export_scipp_simple_keeps_standard_behavior():
    data, state = _make_data()
    events = data.export_scipp_simple(
        source_name="source", sample_name="sample_position"
    )
    assert state["requested_variables"] == ["p", "t", "id"]
    assert events.unit == sc.units.counts
    np.testing.assert_array_equal(events.values, [1.0, 2.0, 3.0])
    assert events.coords["t"].unit == sc.units.s
    np.testing.assert_array_equal(events.coords["t"].values, [10.0, 20.0, 30.0])
    assert "position" in events.coords
    assert "source_position" in events.coords
    assert "sample_position" in events.coords


def test_export_scipp_simple_extra_variables():
    data, state = _make_data()
    events = data.export_scipp_simple(
        source_name="source",
        sample_name="sample_position",
        extra_variables=[
            Transfer(
                mcstas_variable="L", scipp_coord="sim_wavelength", unit="angstrom"
            ),
            Transfer(mcstas_variable="x", scipp_coord="x", unit="m"),
        ],
    )
    assert state["requested_variables"] == ["p", "t", "id", "L", "x"]
    assert events.coords["sim_wavelength"].unit == sc.units.angstrom
    np.testing.assert_array_equal(
        events.coords["sim_wavelength"].values, [1.5, 1.7, 1.9]
    )
    assert events.coords["x"].unit == sc.units.m
    np.testing.assert_array_equal(events.coords["x"].values, [-0.5, 0.0, 0.5])


def test_export_scipp_simple_single_variable():
    data, state = _make_data()
    events = data.export_scipp_simple(
        source_name="source",
        sample_name="sample_position",
        extra_variables=Transfer(mcstas_variable="L", scipp_coord="sim_wavelength"),
    )
    assert state["requested_variables"] == ["p", "t", "id", "L"]
    np.testing.assert_array_equal(
        events.coords["sim_wavelength"].values, [1.5, 1.7, 1.9]
    )


def test_export_scipp_keeps_standard_behavior():
    data, state = _make_data()
    output = data.export_scipp(source_name="source", sample_name="sample_position")
    assert state["requested_variables"] == ["p", "t", "id"]
    events = output["events"]
    assert events.unit == sc.units.counts
    np.testing.assert_array_equal(_flatten_binned(events.data), [1.0, 2.0, 3.0])
    assert set(events.bins.coords) == {"t"}
    assert events.bins.coords["t"].unit == sc.units.s
    assert "position" in events.coords
    assert list(output["bank_names"].values) == ["Square_1"]
    assert output["bank_ids"].values.tolist() == [[0, N_EVENTS - 1]]


def test_export_scipp_extra_variables():
    data, state = _make_data()
    output = data.export_scipp(
        source_name="source",
        sample_name="sample_position",
        extra_variables=[
            Transfer(
                mcstas_variable="L", scipp_coord="sim_wavelength", unit="angstrom"
            ),
            Transfer(mcstas_variable="x", scipp_coord="x", unit="m"),
        ],
    )
    assert state["requested_variables"] == ["p", "t", "id", "L", "x"]
    events = output["events"]
    assert events.bins.coords["sim_wavelength"].unit == sc.units.angstrom
    np.testing.assert_array_equal(
        _flatten_binned(events.bins.coords["sim_wavelength"]), [1.5, 1.7, 1.9]
    )
    assert events.bins.coords["x"].unit == sc.units.m
    np.testing.assert_array_equal(
        _flatten_binned(events.bins.coords["x"]), [-0.5, 0.0, 0.5]
    )


@pytest.mark.parametrize(
    "extra_variables",
    ["L", ("L",), {"L": "sim_wavelength"}, 3.0],
)
def test_extra_variables_rejects_invalid_input(extra_variables):
    data, _ = _make_data()
    with pytest.raises(TypeError):
        data.export_scipp_simple(
            source_name="source",
            sample_name="sample_position",
            extra_variables=extra_variables,
        )
    with pytest.raises(TypeError):
        data.export_scipp(
            source_name="source",
            sample_name="sample_position",
            extra_variables=extra_variables,
        )


def test_extra_variables_rejects_list_with_invalid_entry():
    data, _ = _make_data()
    invalid_list = [
        Transfer(mcstas_variable="x", scipp_coord="x", unit="m"),
        "L",
    ]
    with pytest.raises(TypeError):
        data.export_scipp_simple(
            source_name="source",
            sample_name="sample_position",
            extra_variables=invalid_list,
        )
    with pytest.raises(TypeError):
        data.export_scipp(
            source_name="source",
            sample_name="sample_position",
            extra_variables=["L"],
        )
