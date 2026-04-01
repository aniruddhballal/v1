# tests/test_explainer.py
import numpy as np
import pandas as pd
import pytest
from loader import load
from baseline import learn
from detector import run as detect
from explainer import run as explain, summary


def make_spiked_df(tmp_path, n=200):
    np.random.seed(42)
    f = tmp_path / "data.csv"
    vals = np.random.normal(50, 5, n)
    pd.DataFrame({"value": vals}).to_csv(f, index=False)
    df = load(str(f))
    df.loc[10, "value"] = 999.0   # inject spike
    return df


def test_explanations_returned(tmp_path):
    df = make_spiked_df(tmp_path)
    profile = learn(df)
    result = detect(df, profile)
    explanations = explain(result, profile, df)
    assert len(explanations) > 0


def test_explanation_has_required_fields(tmp_path):
    df = make_spiked_df(tmp_path)
    profile = learn(df)
    result = detect(df, profile)
    explanations = explain(result, profile, df)
    for e in explanations:
        for key in ["row", "anomaly_score", "methods_flagged", "explanation", "flags"]:
            assert key in e


def test_spike_row_has_explanation(tmp_path):
    df = make_spiked_df(tmp_path)
    profile = learn(df)
    result = detect(df, profile)
    explanations = explain(result, profile, df)
    rows = [e["row"] for e in explanations]
    assert 10 in rows


def test_explanation_text_is_readable(tmp_path):
    df = make_spiked_df(tmp_path)
    profile = learn(df)
    result = detect(df, profile)
    explanations = explain(result, profile, df)
    spike = next(e for e in explanations if e["row"] == 10)
    # Should mention the column name and a numeric value
    assert "value" in spike["explanation"]
    assert any(c.isdigit() for c in spike["explanation"])


def test_sorted_by_score(tmp_path):
    df = make_spiked_df(tmp_path)
    profile = learn(df)
    result = detect(df, profile)
    explanations = explain(result, profile, df)
    scores = [e["anomaly_score"] for e in explanations]
    assert scores == sorted(scores, reverse=True)


def test_summary_string(tmp_path):
    df = make_spiked_df(tmp_path)
    profile = learn(df)
    result = detect(df, profile)
    explanations = explain(result, profile, df)
    out = summary(explanations)
    assert "anomalies detected" in out