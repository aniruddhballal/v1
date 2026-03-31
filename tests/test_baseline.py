# tests/test_baseline.py
import numpy as np
import pandas as pd
import pytest
from loader import load
from baseline import learn, save, load_profile

def make_df(tmp_path, rows=100):
    f = tmp_path / "data.csv"
    vals = np.random.normal(50, 10, rows)
    pd.DataFrame({"value": vals}).to_csv(f, index=False)
    return load(str(f))

def test_profile_has_expected_keys(tmp_path):
    df = make_df(tmp_path)
    profile = learn(df)
    col = list(profile["columns"].keys())[0]
    for key in ["mean", "std", "median", "p5", "p95", "q1", "q3", "iqr"]:
        assert key in profile["columns"][col]

def test_profile_mean_reasonable(tmp_path):
    df = make_df(tmp_path, rows=500)
    profile = learn(df)
    # After normalization mean should be in [0, 1]
    mean = profile["columns"]["value"]["mean"]
    assert 0.0 <= mean <= 1.0

def test_save_and_reload(tmp_path):
    df = make_df(tmp_path)
    profile = learn(df)
    path = str(tmp_path / "baseline.json")
    save(profile, path)
    loaded = load_profile(path)
    assert loaded["columns"].keys() == profile["columns"].keys()

def test_timeseries_has_rolling(tmp_path):
    f = tmp_path / "ts.csv"
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=60)
    vals = np.random.normal(100, 5, 60)
    pd.DataFrame({"timestamp": dates, "value": vals}).to_csv(f, index=False)
    df = load(str(f))
    profile = learn(df, window=10)
    assert "rolling_mean" in profile["columns"]["value"]