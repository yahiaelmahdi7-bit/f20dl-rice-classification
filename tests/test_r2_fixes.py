"""Regression tests for the R2 fix round. Slices of stage source run on made-up data; the end-to-end ones run in a temporary copy on synthetic grains."""
import json, os, pathlib, shutil, subprocess, sys
import numpy as np
import pandas as pd
import pytest
import common as C

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _block(rel, start, end):
    src = (ROOT / rel).read_text().split("PROMPT RECORD END ===", 1)[-1]
    i = src.index(start)
    return src[i:src.index(end, i)]


# ---- fix 4: silhouette on several samples
def test_first_silhouette_draw_keeps_its_old_stream():
    ns = {"np": np, "C": C, "len": len}
    exec(_block("src/04_cluster.py", "rng = np.random", "SETS ="), ns)
    body = _block("src/04_cluster.py", "    sidx = rng.choice", "    lab6 = {}")
    old = np.random.default_rng(C.SEED)
    ns["Z"] = np.zeros((20000, 1))
    for _ in range(3):                 # the second and third set are the ones extra draws would shift
        exec("if 1:\n" + body, ns)
        assert np.array_equal(ns["sidx"], old.choice(20000, 10000, replace=False))
        assert len(ns["draws"]) == ns["N_SIL"] > 1 and ns["draws"][0] is ns["sidx"]
        assert not np.array_equal(ns["draws"][1], ns["sidx"])


def test_silhouette_row_keeps_first_draw_and_adds_spread():
    vals = iter([0.5, 0.1, 0.3])
    ns = {"np": np, "silhouette_score": lambda a, b: next(vals)}
    exec(_block("src/04_cluster.py", "def sil_draws(", "for fs in SETS:"), ns)
    r = ns["sil_draws"](np.zeros(4), np.zeros(4), [0, 1, 2])
    assert r["silhouette"] == 0.5 and r["silhouette_mean"] == pytest.approx(0.3) and r["silhouette_sd"] == pytest.approx(0.2) and r["silhouette_draws"] == 3
