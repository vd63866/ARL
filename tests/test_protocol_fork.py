"""Regression coverage for paired policy fork isolation."""

from __future__ import annotations

import pytest

from adaptive_rl.protocol.fork import fork_adaptive_and_fixed, model_fingerprint

torch = pytest.importorskip("torch")


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)


class _Algorithm:
    def __init__(self) -> None:
        self.model = _Model()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def predict(self, observation, deterministic=True):
        del deterministic
        with torch.no_grad():
            return self.model(torch.as_tensor(observation, dtype=torch.float32)), None

    def train(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("fixed facade must not expose training")


def test_fork_is_equal_at_start_and_independently_mutable() -> None:
    source = _Algorithm()
    expected = model_fingerprint(source)
    adaptive, fixed, frozen_fingerprint = fork_adaptive_and_fixed(source)

    assert expected == frozen_fingerprint
    assert model_fingerprint(adaptive) == fixed.fingerprint == expected
    assert adaptive.model.layer.weight.data_ptr() != source.model.layer.weight.data_ptr()
    assert (
        adaptive.model.layer.weight.data_ptr()
        != fixed._FrozenPolicy__algorithm.model.layer.weight.data_ptr()
    )
    assert adaptive.optimizer is not source.optimizer
    assert fixed._FrozenPolicy__algorithm.optimizer is not source.optimizer

    with torch.no_grad():
        adaptive.model.layer.weight.add_(1.0)
    assert model_fingerprint(adaptive) != expected
    assert fixed.fingerprint == expected

    with pytest.raises(AttributeError):
        fixed.train(10)


def test_model_fingerprint_is_deterministic_and_sensitive_to_buffers() -> None:
    model = _Model()
    initial = model_fingerprint(model)
    assert model_fingerprint(model) == initial
    with torch.no_grad():
        model.layer.bias[0].add_(0.25)
    assert model_fingerprint(model) != initial
