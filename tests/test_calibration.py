import numpy as np
import pytest

from opendecisions.calibration import Calibrator, expected_calibration_error, fit_platt, fit_temperature


def _synthetic(n=3000, k=4, true_t=2.5, seed=0):
    """Logits that are over-confident by a factor true_t: labels drawn from softmax(z / true_t)."""
    rng = np.random.default_rng(seed)
    logits, labels = [], []
    for _ in range(n):
        z = rng.normal(0, 4, size=k)
        p = np.exp(z / true_t - (z / true_t).max())
        p /= p.sum()
        logits.append(z.tolist())
        labels.append(int(rng.choice(k, p=p)))
    return logits, labels


def test_temperature_recovers_true_value():
    logits, labels = _synthetic()
    assert fit_temperature(logits, labels) == pytest.approx(2.5, rel=0.12)


def test_temperature_reduces_ece():
    logits, labels = _synthetic()
    cal = Calibrator.fit([("choice", z, y) for z, y in zip(logits, labels)])

    def ece(c):
        ps = [c.apply("choice", z) for z in logits]
        return expected_calibration_error([p.max() for p in ps], [int(np.argmax(p)) == y for p, y in zip(ps, labels)])

    assert ece(cal) < ece(Calibrator()) / 2


def test_platt_fixes_yes_bias():
    rng = np.random.default_rng(1)
    margins = rng.normal(0, 3, 4000)
    # the model says "yes" 2 nats too eagerly
    y = (rng.random(4000) < 1 / (1 + np.exp(-(margins - 2.0)))).astype(int)
    a, b = fit_platt(margins.tolist(), y.tolist())
    assert a == pytest.approx(1.0, abs=0.15) and b == pytest.approx(-2.0, abs=0.25)


def test_calibrator_roundtrip(tmp_path):
    cal = Calibrator(temperature={"choice": 1.7}, platt=(0.9, -0.3))
    path = tmp_path / "c.json"
    cal.save(path)
    back = Calibrator.load(path)
    assert back.temperature == {"choice": 1.7} and back.platt == (0.9, -0.3)
    p = back.apply("noul", [0.0, 0.0])
    assert p[0] == pytest.approx(1 / (1 + np.exp(0.3)))
