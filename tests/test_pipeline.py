from pathlib import Path

import numpy as np
import pytest

from src.preprocess import clean_text

ARTIFACTS = Path("artifacts/v0")
needs_model = pytest.mark.skipif(not (ARTIFACTS / "model.pt").exists(),
                                 reason="run `python -m src.train` first")

SAMPLES = [
    "<br />This movie was absolutely wonderful, I loved every minute!",
    "Terrible plot, awful acting. I couldn't wait for it to end.",
    "",
]


def test_clean_text_fixed_string():
    assert clean_text("<br />I didn't LOVE it!! 10/10 http://x.com") == "i did not love it"


@needs_model
def test_predict_proba_shape_and_range():
    from src.predict import load_model, predict_proba
    model, vocab = load_model(ARTIFACTS)
    probs = predict_proba(SAMPLES, model, vocab)
    assert probs.shape == (len(SAMPLES),)
    assert np.all((probs >= 0) & (probs <= 1))


@needs_model
def test_reload_is_deterministic():
    from src.predict import load_model, predict_proba
    first = predict_proba(SAMPLES, *load_model(ARTIFACTS))
    second = predict_proba(SAMPLES, *load_model(ARTIFACTS))
    assert np.array_equal(first, second)
