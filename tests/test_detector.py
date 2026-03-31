# tests/test_detector.py
import numpy as np
import pandas as pd
import pytest
from loader import load
from baseline import learn
from detector import run


def make_clean_df(tmp_path, n=200):
    np.random.seed(42)
    f = tmp_path / "clean.csv"
    pd.DataFrame({"value": np.random.normal(50, 5, n)}).to_csv(f, index=False)
    return load(str(f))


def test_anomaly_columns_present(tmp_path):
    df = make_clean_df(tmp_path)
    profile = learn(df)
    result = run(df, profile)
    for col in ["flag_zscore", "flag_iqr", "flag_isoforest", "anomaly_score", "methods_flagged"]:
        assert col in result.columns


def test_injected_spike_detected(tmp_path):
    df = make_clean_df(tmp_path, n=300)
    profile = learn(df)
    # Inject obvious spike at row 50
    df_spiked = df.copy()
    df_spiked.loc[50, "value"] = 999.0
    result = run(df_spiked, profile)
    assert result.loc[50, "flag_zscore"] or result.loc[50, "flag_iqr"]


def test_clean_data_low_false_positives(tmp_path):
    df = make_clean_df(tmp_path, n=500)
    profile = learn(df)
    result = run(df, profile)
    # On clean normal data, fewer than 5% flagged by statistical methods
    flagged_rate = result["flag_zscore"].mean()
    assert flagged_rate < 0.05


def test_rule_based_detection(tmp_path):
    df = make_clean_df(tmp_path, n=100)
    profile = learn(df)
    # After normalization values are in [0,1], so rule gt=0.99 should catch the max
    rules = {"value": {"gt": 0.99}}
    result = run(df, profile, rules=rules)
    assert result["flag_rules"].any()


def test_anomaly_score_bounded(tmp_path):
    df = make_clean_df(tmp_path, n=200)
    profile = learn(df)
    result = run(df, profile)
    assert result["anomaly_score"].between(0.0, 1.0).all()