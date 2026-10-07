# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)
"""Tests for chunked Nexus loading and Scipp export."""

import h5py
import numpy as np
import pytest

from mcstastox.LoadFile import Data, Variable


def _write_event_component(components, index, name, pixel_ids, position, events):
    component = components.create_group(f"{index:04d}_{name}")
    component.create_dataset("Position", data=position)
    component.create_dataset("Rotation", data=np.eye(3))

    geometry = component.create_group("Geometry")
    geometry.attrs["xmin"] = np.bytes_("-1.0")
    geometry.attrs["xmax"] = np.bytes_("1.0")
    geometry.attrs["ymin"] = np.bytes_("-1.0")
    geometry.attrs["ymax"] = np.bytes_("1.0")
    geometry.attrs["Shape identifier"] = np.bytes_("0")

    output = component.create_group("output")
    bins = output.create_group("BINS")
    bins.attrs["xvar"] = np.bytes_("x")
    bins.attrs["xlabel"] = np.bytes_("x")
    bins.attrs["yvar"] = np.bytes_("y")
    bins.attrs["ylabel"] = np.bytes_("y")
    bins.create_dataset("x", data=np.linspace(-0.5, 0.5, 4))
    bins.create_dataset("y", data=np.linspace(-0.5, 0.5, 3))
    bins.create_dataset("pixels", data=pixel_ids.reshape(3, 4))

    event_data = output.create_group("event_data")
    event_data.attrs["variables"] = np.bytes_("p t id L")
    event_data.create_dataset("events", data=events)


def _write_fixture(file_path):
    pixel_ids = np.arange(12, dtype=int)
    weights = np.array([0.0, *np.ones(11)])
    times = np.arange(12, dtype=float)
    wavelengths = np.linspace(1.0, 2.0, 12)
    events = np.column_stack((weights, times, pixel_ids, wavelengths))

    with h5py.File(file_path, "w") as file_handle:
        entry = file_handle.create_group("entry1")
        entry.create_group("data")

        simulation = entry.create_group("simulation")
        simulation.attrs["program"] = np.bytes_("3.6.16")
        simulation.create_group("Param")

        instrument = entry.create_group("instrument")
        components = instrument.create_group("components")
        for index, name, component_ids, position in (
            (0, "source", None, [0.0, 0.0, -1.0]),
            (1, "sample_position", None, [0.0, 0.0, 0.0]),
            (2, "Square_1", pixel_ids, [0.0, 0.0, 1.0]),
            (3, "Square_2", pixel_ids + 12, [1.0, 0.0, 1.0]),
        ):
            if component_ids is None:
                component = components.create_group(f"{index:04d}_{name}")
                component.create_dataset("Position", data=position)
                component.create_dataset("Rotation", data=np.eye(3))
            else:
                _write_event_component(
                    components,
                    index,
                    name,
                    component_ids,
                    position,
                    events,
                )


@pytest.fixture(scope="module")
def nexus_fixture(tmp_path_factory):
    data_folder = tmp_path_factory.mktemp("nexus")
    _write_fixture(data_folder / "mccode.h5")
    return data_folder


def _flatten_binned(array):
    """Flatten a binned Scipp variable to a single NumPy array."""
    values = [np.asarray(value.values) for value in array.values]
    return np.concatenate(values) if values else np.empty(0)


def test_event_iterator_matches_full_read(nexus_fixture):
    variables = ["p", "t", "id", "L"]
    with Data(nexus_fixture) as data:
        full = data.get_event_data(
            variables,
            component_name="Square_1",
            filter_zeros=False,
        )
        chunks = {var: [] for var in variables}
        for event_data in data.file_object.iter_event_data(
            variables, component_name="Square_1", chunk_size=7
        ):
            for var in variables:
                chunks[var].append(event_data[var])

    for variable in variables:
        iterated = np.concatenate(chunks[variable])
        np.testing.assert_array_equal(iterated, full[variable])


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_simple_export_chunking_matches_full_export(chunk_size, nexus_fixture):
    extra = Variable("wavelength", "L", "angstrom")
    with Data(nexus_fixture) as data:
        full = data.export_scipp_simple(
            "source",
            "sample_position",
            component_name="Square_1",
            extra_variables=extra,
        )
        chunked = data.export_scipp_simple(
            "source",
            "sample_position",
            component_name="Square_1",
            extra_variables=extra,
            chunk_size=chunk_size,
        )

    np.testing.assert_array_equal(chunked.values, full.values)
    for coord in ("position", "t", "wavelength"):
        np.testing.assert_array_equal(
            chunked.coords[coord].values, full.coords[coord].values
        )


@pytest.mark.parametrize("filter_zeros", [True, False])
def test_grouped_export_chunking_matches_full_export(filter_zeros, nexus_fixture):
    with Data(nexus_fixture) as data:
        full = data.export_scipp(
            "source",
            "sample_position",
            component_name="Square_1",
            filter_zeros=filter_zeros,
        )
        chunked = data.export_scipp(
            "source",
            "sample_position",
            component_name="Square_1",
            filter_zeros=filter_zeros,
            chunk_size=7,
        )

    np.testing.assert_array_equal(
        _flatten_binned(chunked["events"].data),
        _flatten_binned(full["events"].data),
    )
    np.testing.assert_array_equal(
        _flatten_binned(chunked["events"].bins.coords["t"]),
        _flatten_binned(full["events"].bins.coords["t"]),
    )
    np.testing.assert_array_equal(chunked["bank_ids"].values, full["bank_ids"].values)
    np.testing.assert_array_equal(
        chunked["bank_names"].values, full["bank_names"].values
    )


def test_event_iterator_defaults_to_all_components(nexus_fixture):
    variables = ["p", "t", "id"]
    with Data(nexus_fixture) as data:
        full = data.file_object.get_event_data(variables)
        chunks = {var: [] for var in variables}
        for event_data in data.file_object.iter_event_data(variables, chunk_size=7):
            for var in variables:
                chunks[var].append(event_data[var])

    for variable in variables:
        iterated = np.concatenate(chunks[variable])
        np.testing.assert_array_equal(iterated, full[variable])


def test_event_iterator_requires_chunk_size(nexus_fixture):
    with Data(nexus_fixture) as data:
        with pytest.raises(TypeError, match="chunk_size"):
            data.file_object.iter_event_data(["p"])


def test_simple_export_chunking_handles_empty_result():
    data = Data.__new__(Data)
    data.component_pixel_order = ["Square_1"]
    data.pixel_range = {"Square_1": [0, 0]}
    data._iter_event_chunks = lambda *args: iter(())
    data.get_id_to_global_coordinates = lambda component_name=None: np.zeros((1, 3))
    data.get_global_component_coordinates = lambda component_name: np.zeros(3)

    events = data.export_scipp_simple(
        "source", "sample", component_name="Square_1", chunk_size=2
    )

    assert events.sizes["events"] == 0


@pytest.mark.parametrize("chunk_size", [None, 0, -1, 1.5, True, "4"])
def test_chunk_size_must_be_positive_integer(chunk_size, nexus_fixture):
    with Data(nexus_fixture) as data:
        with pytest.raises((TypeError, ValueError)):
            list(data.file_object.iter_event_data(["p"], chunk_size=chunk_size))
