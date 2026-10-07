# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)
"""Tests for chunked Nexus loading and Scipp export."""

from pathlib import Path

import numpy as np
import pytest

from mcstastox.LoadFile import Data, Variable

FIXTURE = Path(__file__).parents[1] / "docs" / "user-guide" / "test_12"


def _flatten_binned(array):
    """Flatten a binned Scipp variable to a single NumPy array."""
    values = [np.asarray(value.values) for value in array.values]
    return np.concatenate(values) if values else np.empty(0)


def test_event_iterator_matches_full_read():
    variables = ["p", "t", "id", "L"]
    with Data(FIXTURE) as data:
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
def test_simple_export_chunking_matches_full_export(chunk_size):
    extra = Variable("wavelength", "L", "angstrom")
    with Data(FIXTURE) as data:
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
def test_grouped_export_chunking_matches_full_export(filter_zeros):
    with Data(FIXTURE) as data:
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


def test_event_iterator_defaults_to_all_components():
    variables = ["p", "t", "id"]
    with Data(FIXTURE) as data:
        full = data.file_object.get_event_data(variables)
        chunks = {var: [] for var in variables}
        for event_data in data.file_object.iter_event_data(variables, chunk_size=7):
            for var in variables:
                chunks[var].append(event_data[var])

    for variable in variables:
        iterated = np.concatenate(chunks[variable])
        np.testing.assert_array_equal(iterated, full[variable])


def test_event_iterator_requires_chunk_size():
    with Data(FIXTURE) as data:
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
def test_chunk_size_must_be_positive_integer(chunk_size):
    with Data(FIXTURE) as data:
        with pytest.raises((TypeError, ValueError)):
            list(data.file_object.iter_event_data(["p"], chunk_size=chunk_size))
