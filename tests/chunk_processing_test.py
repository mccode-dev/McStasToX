# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)
"""Tests for chunked Nexus loading and Scipp export."""

from pathlib import Path

import numpy as np
import pytest

from mcstastox import SamplingSettings
from mcstastox.LoadFile import Data, Transfer

FIXTURE = Path(__file__).parents[1] / "docs" / "user-guide" / "test_12"


def _flatten_binned(array):
    """Flatten a binned Scipp variable to a single NumPy array."""
    values = [np.asarray(value.values) for value in array.values]
    return np.concatenate(values) if values else np.empty(0)


def test_get_event_data_chunking_matches_full_read():
    with Data(FIXTURE) as data:
        full = data.get_event_data(
            ["p", "t", "id", "L"],
            component_name="Square_1",
            filter_zeros=False,
        )
        chunked = data.get_event_data(
            ["p", "t", "id", "L"],
            component_name="Square_1",
            filter_zeros=False,
            chunk_size=7,
        )
        nexus_full = data.file_object.get_event_data(
            ["p", "t", "id"], component_name="Square_1"
        )
        nexus_chunked = data.file_object.get_event_data(
            ["p", "t", "id"], component_name="Square_1", chunk_size=7
        )

    for variable in full:
        np.testing.assert_array_equal(chunked[variable], full[variable])
    for variable in nexus_full:
        np.testing.assert_array_equal(nexus_chunked[variable], nexus_full[variable])


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_simple_export_chunking_matches_full_export(chunk_size):
    extra = Transfer("L", "wavelength", "angstrom")
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


def test_chunking_supports_multiple_components():
    with Data(FIXTURE) as data:
        full = data.get_event_data(["p", "t", "id"], filter_zeros=False)
        chunked = data.get_event_data(
            ["p", "t", "id"], filter_zeros=False, chunk_size=7
        )

    for variable in full:
        np.testing.assert_array_equal(chunked[variable], full[variable])


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


def test_simple_export_sampling_reads_in_chunks():
    with Data(FIXTURE) as data:
        events = data.export_scipp_simple(
            "source",
            "sample_position",
            component_name="Square_1",
            chunk_size=7,
            sampling=SamplingSettings(n_samples=25, seed=42),
        )

    assert events.sizes["events"] == 25
    np.testing.assert_array_equal(events.values, np.ones(25))


def test_grouped_export_sampling_returns_unit_weight_events():
    with Data(FIXTURE) as data:
        output = data.export_scipp(
            "source",
            "sample_position",
            component_name="Square_1",
            chunk_size=7,
            sampling=SamplingSettings(n_samples=25, seed=42),
        )

    np.testing.assert_array_equal(_flatten_binned(output["events"].data), np.ones(25))


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, "4"])
def test_chunk_size_must_be_positive_integer(chunk_size):
    with Data(FIXTURE) as data:
        with pytest.raises((TypeError, ValueError)):
            data.get_event_data(["p"], chunk_size=chunk_size)
