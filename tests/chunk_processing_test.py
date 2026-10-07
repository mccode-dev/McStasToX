# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Mccode-dev contributors (https://github.com/mccode-dev)
"""Tests for chunked Nexus loading and Scipp export."""

from pathlib import Path

import mcstasscript as ms
import numpy as np
import pytest

from mcstastox import SamplingSettings
from mcstastox.LoadFile import Data, Transfer


def _make_test_instrument() -> ms.McStas_instr:
    """Build the small McStas instrument used by the chunking tests."""
    instr = ms.McStas_instr("chunk_processing_test")
    source = instr.add_component("source", "Source_simple")
    source.set_parameters(
        xwidth=0.01,
        yheight=0.01,
        focus_xw=0.01,
        focus_yh=0.01,
        dist=2,
        lambda0=instr.add_parameter("wavelength", value=1.8),
        dlambda=instr.add_parameter("delta_wavelength", value=1.3),
    )

    sample_position = instr.add_component("sample_position", "Arm")
    sample_position.set_AT(source.dist, RELATIVE=source)
    sample = instr.add_component("sample", "PowderN", RELATIVE=sample_position)
    sample.set_parameters(
        radius=source.xwidth / 2,
        yheight=source.focus_yh,
        reflections='"Cu.laz"',
    )

    detector_direction = instr.add_component(
        "detector_direction_square_1",
        "Arm",
        RELATIVE=sample_position,
        ROTATED=[0, -140, 0],
    )
    monitor = instr.add_component("Square_1", "Monitor_nD")
    monitor.set_parameters(
        xwidth=0.1,
        yheight=0.1,
        restore_neutron=1,
        filename='"direct_event_square_signal.dat"',
    )
    monitor.options = (
        '"mantid square x bins=15 y bins=15, '
        'neutron pixel min=0 t, l, list all neutrons"'
    )
    monitor.set_AT(0.35, RELATIVE=detector_direction)

    detector_direction = instr.add_component(
        "detector_direction_square_2",
        "Arm",
        RELATIVE=sample_position,
        ROTATED=[20, 57, 0],
    )
    monitor = instr.add_component("Square_2", "Monitor_nD")
    monitor.set_parameters(
        xwidth=0.25,
        yheight=0.1,
        restore_neutron=1,
        filename='"scattered_event_square_signal.dat"',
    )
    monitor.options = (
        '"mantid square x bins=30 y bins=15, '
        'neutron pixel min=225 t, list all neutrons"'
    )
    monitor.set_AT(0.5, RELATIVE=detector_direction)

    return instr


@pytest.fixture(scope="session")
def data_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a temporary NeXus event file with McStas."""
    work_folder = tmp_path_factory.mktemp("chunk_processing")
    output_path = work_folder / "results"
    instr = _make_test_instrument()
    instr.set_parameters(wavelength=1.8, delta_wavelength=1.3)
    instr.settings(
        ncount=10_000,
        output_path=str(output_path),
        suppress_output=True,
        NeXus=True,
    )
    instr.backengine()
    if not (output_path / "mccode.h5").exists():
        raise RuntimeError("McStas did not generate the test NeXus file")
    return output_path


def _flatten_binned(array):
    """Flatten a binned Scipp variable to a single NumPy array."""
    values = [np.asarray(value.values) for value in array.values]
    return np.concatenate(values) if values else np.empty(0)


def test_get_event_data_chunking_matches_full_read(data_folder):
    with Data(data_folder) as data:
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
def test_simple_export_chunking_matches_full_export(data_folder, chunk_size):
    extra = Transfer("L", "wavelength", "angstrom")
    with Data(data_folder) as data:
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
def test_grouped_export_chunking_matches_full_export(data_folder, filter_zeros):
    with Data(data_folder) as data:
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


def test_chunking_supports_multiple_components(data_folder):
    with Data(data_folder) as data:
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


def test_simple_export_sampling_reads_in_chunks(data_folder):
    with Data(data_folder) as data:
        events = data.export_scipp_simple(
            "source",
            "sample_position",
            component_name="Square_1",
            chunk_size=7,
            sampling=SamplingSettings(n_samples=25, seed=42),
        )

    assert events.sizes["events"] == 25
    np.testing.assert_array_equal(events.values, np.ones(25))


def test_grouped_export_sampling_returns_unit_weight_events(data_folder):
    with Data(data_folder) as data:
        output = data.export_scipp(
            "source",
            "sample_position",
            component_name="Square_1",
            chunk_size=7,
            sampling=SamplingSettings(n_samples=25, seed=42),
        )

    np.testing.assert_array_equal(_flatten_binned(output["events"].data), np.ones(25))


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, "4"])
def test_chunk_size_must_be_positive_integer(data_folder, chunk_size):
    with Data(data_folder) as data:
        with pytest.raises((TypeError, ValueError)):
            data.get_event_data(["p"], chunk_size=chunk_size)
