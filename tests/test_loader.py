# tests/test_loader.py
import pandas as pd
import numpy as np
import pytest
from loader import load

def test_csv_loads(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("a,b,c\n1,2,3\n4,,6\n7,8,9")
    df = load(str(f))
    assert not df.isnull().any().any()  # no missing values
    assert len(df) == 3

def test_normalization(tmp_path):
    f = tmp_path / "norm.csv"
    f.write_text("val\n0\n50\n100")
    df = load(str(f))
    assert df["val"].max() <= 1.0
    assert df["val"].min() >= 0.0

def test_timeseries_detection(tmp_path):
    f = tmp_path / "ts.csv"
    f.write_text("timestamp,value\n2024-01-01,10\n2024-01-02,20")
    df = load(str(f))
    assert df.attrs["is_timeseries"] is True