# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Mccode-dev contributors (https://github.com/mccode-dev)
"""Tests for weighted event sampling."""

import numpy as np
import pytest

from mcstastox import SamplingSettings, sample_event_chunks


def test_sampling_is_weighted_and_returns_unit_weights():
    result = sample_event_chunks(
        [
            {
                "p": np.array([1.0, 3.0]),
                "id": np.array([10, 20]),
            }
        ],
        SamplingSettings(n_samples=1000, seed=42),
    )

    assert result["p"].shape == (1000,)
    np.testing.assert_array_equal(result["p"], np.ones(1000))
    assert 0.65 < np.mean(result["id"] == 20) < 0.85
    assert result.total_weight == 4.0
    assert result.total_weight_variance == 10.0
    assert result.effective_duration == 250.0
    assert result.effective_duration_variance == 39062.5


def test_sampling_handles_chunks_and_ordering():
    result = sample_event_chunks(
        [
            {"p": np.array([1.0, 0.0]), "id": np.array([0, 1])},
            {"p": np.array([1.0]), "id": np.array([2])},
        ],
        SamplingSettings(n_samples=20, seed=3, ordered=True),
    )

    assert np.all(np.diff(result["id"]) >= 0)
    assert set(result["id"]).issubset({0, 2})


def test_sampling_with_no_positive_weights_is_empty():
    result = sample_event_chunks(
        [{"p": np.zeros(3), "id": np.array([1, 2, 3])}],
        SamplingSettings(n_samples=5, seed=1),
    )

    assert result["p"].size == 0
    assert result["id"].size == 0
    assert result.effective_duration is None


@pytest.mark.parametrize("kwargs", [{"n_samples": 0}, {"n_samples": -1}])
def test_sampling_settings_reject_invalid_sample_size(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        SamplingSettings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 1, "seed": True},
        {"n_samples": 1, "ordered": 1},
    ],
)
def test_sampling_settings_reject_invalid_field_types(kwargs):
    with pytest.raises(TypeError):
        SamplingSettings(**kwargs)


def test_sampling_rejects_negative_weights():
    with pytest.raises(ValueError, match="not be negative"):
        sample_event_chunks(
            [{"p": np.array([1.0, -1.0]), "id": np.array([1, 2])}],
            SamplingSettings(n_samples=2),
        )
