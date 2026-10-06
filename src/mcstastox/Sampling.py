# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Mccode-dev contributors (https://github.com/mccode-dev)
"""Streaming weighted event sampling utilities.

The reservoir update is adapted from the WRSWRSKIP algorithm used by
`stream_sampling_mcstas <https://github.com/aaronfinke/stream_sampling_mcstas>`_.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def _sample_positive_binomial(
    rng: np.random.Generator, n: int, probability: float
) -> int:
    """Sample ``Binomial(n, probability)`` conditioned on being positive."""
    if probability >= 1.0:
        return n

    zero_probability = (1.0 - probability) ** n
    if zero_probability < 0.5:
        count = 0
        while count == 0:
            count = int(rng.binomial(n, probability))
        return count

    # Rejection is slow when a positive draw is rare. Invert the small tail
    # instead, using unnormalized probabilities to avoid a second division.
    if zero_probability == 1.0:
        return 1
    target = rng.random() * (1.0 - zero_probability)
    mass = n * probability * (1.0 - probability) ** (n - 1)
    cumulative = mass
    count = 1
    while target > cumulative and count < n:
        mass *= (n - count) / (count + 1) * probability / (1.0 - probability)
        count += 1
        cumulative += mass
    return count


@dataclass(frozen=True)
class SamplingSettings:
    """Settings for converting weighted events to normal events.

    Sampling uses weighted reservoir sampling with replacement. This means
    that every sampled event has unit weight and that an input event may occur
    more than once in the result.

    :param n_samples: Number of normal events to produce
    :param seed: Optional seed for reproducible sampling
    :param ordered: If True, return sampled events in input stream order
    """

    n_samples: int
    seed: int | None = None
    ordered: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.n_samples, bool) or not isinstance(
            self.n_samples, (int, np.integer)
        ):
            raise TypeError("n_samples must be a positive integer")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer))
        ):
            raise TypeError("seed must be an integer or None")
        if not isinstance(self.ordered, bool):
            raise TypeError("ordered must be a bool")


class SampledEventData(dict[str, np.ndarray]):
    """Sampled event arrays and the total weight of the input stream."""

    def __init__(
        self,
        event_data: dict[str, np.ndarray],
        total_weight: float,
        total_weight_variance: float,
    ):
        super().__init__(event_data)
        self.total_weight = total_weight
        self.total_weight_variance = total_weight_variance

    @property
    def effective_duration(self) -> float | None:
        """Return the duration represented by the unit-weight sample in seconds.

        This is defined when the input weights represent a count rate. It is
        ``number of sampled events / total input weight``.
        """
        n_events = len(self.get("p", ()))
        if n_events == 0 or self.total_weight <= 0:
            return None
        return n_events / self.total_weight

    @property
    def effective_duration_variance(self) -> float | None:
        """Return the propagated variance of :attr:`effective_duration`."""
        duration = self.effective_duration
        if duration is None:
            return None
        return duration**2 * self.total_weight_variance / self.total_weight**2


class _WeightedEventSampler:
    """Weighted reservoir sampler for dictionaries of event arrays."""

    def __init__(self, settings: SamplingSettings):
        self._n_samples = settings.n_samples
        self._rng = np.random.default_rng(settings.seed)
        self._ordered = settings.ordered
        self._total_weight = 0.0
        self._total_weight_variance = 0.0
        self._skip_weight = 0.0
        self._seen_events = 0
        self._reservoir: dict[str, np.ndarray] | None = None
        self._positions: np.ndarray | None = None
        self._keys: tuple[str, ...] | None = None
        self._has_samples = False

    def _initialize(self, event_data: dict[str, np.ndarray]) -> None:
        self._keys = tuple(event_data)
        self._reservoir = {
            key: np.empty(self._n_samples, dtype=np.asarray(values).dtype)
            for key, values in event_data.items()
        }
        if self._ordered:
            self._positions = np.empty(self._n_samples, dtype=np.int64)

    def _replace(
        self, event_data: dict[str, np.ndarray], index: int, count: int
    ) -> None:
        if self._reservoir is None or self._keys is None:
            self._initialize(event_data)

        if count == 0:
            return

        slots = self._rng.choice(self._n_samples, size=count, replace=False)
        for key in self._keys:
            self._reservoir[key][slots] = event_data[key][index]
        if self._positions is not None:
            self._positions[slots] = self._seen_events + index
        self._has_samples = True

    def add(self, event_data: dict[str, np.ndarray]) -> None:
        """Consume one chunk of event data."""
        if "p" not in event_data:
            raise ValueError("Sampled event data must contain a 'p' weight array")

        arrays = {key: np.asarray(values) for key, values in event_data.items()}
        weights = arrays["p"]
        if weights.ndim != 1:
            raise ValueError("Event arrays must be one-dimensional")
        if any(values.ndim != 1 for values in arrays.values()):
            raise ValueError("Event arrays must be one-dimensional")
        if any(len(values) != len(weights) for values in arrays.values()):
            raise ValueError("Event arrays must have the same length")
        if not np.all(np.isfinite(weights)):
            raise ValueError("Event weights must be finite")
        if np.any(weights < 0):
            raise ValueError("Event weights must not be negative")

        if self._reservoir is None:
            self._initialize(arrays)

        positive = np.flatnonzero(weights > 0)
        if positive.size:
            positive_weights = weights[positive].astype(np.float64, copy=False)
            cumulative = np.cumsum(positive_weights, dtype=np.float64)
            previous_weight = self._total_weight
            self._total_weight_variance += float(
                np.dot(positive_weights, positive_weights)
            )

            for offset, index in enumerate(positive):
                event_weight = float(positive_weights[offset])
                event_total = previous_weight + float(cumulative[offset])
                if self._skip_weight <= event_total:
                    probability = event_weight / event_total
                    count = _sample_positive_binomial(
                        self._rng, self._n_samples, probability
                    )
                    self._replace(arrays, int(index), count)
                    self._skip_weight = event_total * np.exp(
                        self._rng.exponential() / self._n_samples
                    )

            self._total_weight = previous_weight + float(cumulative[-1])

        self._seen_events += len(weights)

    def result(self) -> SampledEventData:
        """Return sampled event arrays with unit weights."""
        if self._reservoir is None or self._keys is None:
            return SampledEventData({}, self._total_weight, self._total_weight_variance)

        result = {key: values.copy() for key, values in self._reservoir.items()}
        if not self._has_samples:
            return SampledEventData(
                {key: values[:0] for key, values in result.items()},
                self._total_weight,
                self._total_weight_variance,
            )
        if self._positions is not None:
            order = np.argsort(self._positions, kind="stable")
            result = {key: values[order] for key, values in result.items()}
        result["p"] = np.ones(self._n_samples, dtype=np.float64)
        return SampledEventData(result, self._total_weight, self._total_weight_variance)


def sample_event_chunks(
    event_chunks: Iterable[dict[str, np.ndarray]], settings: SamplingSettings
) -> SampledEventData:
    """Sample weighted event dictionaries without collecting the input stream.

    The input dictionaries must contain a one-dimensional ``p`` array and all
    other arrays must have the same length. Only the sampled result is kept in
    memory.

    :param event_chunks: Iterable of event dictionaries
    :param settings: Sampling configuration
    :return: sampled event data with unit ``p`` values and input weight metadata
    """
    if not isinstance(settings, SamplingSettings):
        raise TypeError("settings must be a SamplingSettings instance")

    sampler = _WeightedEventSampler(settings)
    for event_data in event_chunks:
        sampler.add(event_data)
    return sampler.result()
