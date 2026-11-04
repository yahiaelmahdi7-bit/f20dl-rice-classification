"""Planted faults for the shuffled-label controls. Made-up grains only: no real data is read and nothing is written to results/."""
import importlib, sys
import numpy as np, pandas as pd, pytest
import common as C

B = importlib.import_module("05_baselines")
N = importlib.import_module("06_neural")


def grains(n, seed):
    """Five tight clusters, like the real classes: a model given true labels scores near 100%."""
    cols = C.features("all106")
    y = np.arange(n) % 5
    centres = np.random.default_rng(7).normal(0, 2, (5, len(cols)))
    return pd.DataFrame(centres[y] + np.random.default_rng(seed).normal(0, 1, (n, len(cols))), columns=cols).assign(y=y)


class NoShuffle:
    """Fault 1: the shuffle hands the labels back untouched."""
    def permutation(self, y):
        return y


@pytest.fixture
def stage(tmp_path, monkeypatch):
    data = {"train": grains(1500, 1), "val": grains(500, 2)}             # no "test" key: a test read would crash
    monkeypatch.setattr(C, "SMOKE", False)
    monkeypatch.setattr(C, "load_split", lambda name: data[name])
    monkeypatch.setattr(C, "stage_dir", lambda name: tmp_path)
    monkeypatch.setattr(C, "MODELS", tmp_path / "models")
    monkeypatch.setattr(C, "wrote", lambda *a, **k: None)
    monkeypatch.setattr(N, "EP_MLP", 12)
    return tmp_path


def run(mod, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["stage", "--control", *args])
    mod.main()


STAGES = [(B, ["--only", "nb:all106"]), (N, ["--only", "mlp"])]
IDS = ["stage5", "stage6_mlp"]


@pytest.mark.parametrize("mod,args", STAGES, ids=IDS)
def test_clean_control_passes(stage, monkeypatch, mod, args):
    run(mod, monkeypatch, *args)
    assert pd.read_csv(stage / "control.csv").val_accuracy.iloc[0] < C.CONTROL_CEILING


# Fault 1: the shuffle does nothing. The run stops before any model is trained.
@pytest.mark.parametrize("mod,args", STAGES, ids=IDS)
def test_shuffle_that_does_nothing_stops_the_run(stage, monkeypatch, mod, args):
    monkeypatch.setattr(np.random, "default_rng", lambda *a: NoShuffle())
    with pytest.raises(SystemExit, match="100.00% of labels unchanged"):
        run(mod, monkeypatch, *args)
    assert not (stage / "control.csv").exists()


def test_cnn_shuffle_that_does_nothing_stops_the_run(stage, monkeypatch):
    tr = grains(100, 3)
    monkeypatch.setattr(np.random, "default_rng", lambda *a: NoShuffle())
    with pytest.raises(SystemExit, match="100.00% of labels unchanged"):
        N.train_cnn(N.DEFAULT_CNN, 0, tr, None, 1, 1, "t", shuffle=True, stats=(0.0, 1.0))


# Fault 2: the real labels reach the model. The score gives it away, and the row stays on file as evidence.
@pytest.mark.parametrize("mod,args", STAGES, ids=IDS)
def test_real_labels_reaching_the_model_stops_the_run(stage, monkeypatch, mod, args):
    monkeypatch.setattr(C, "shuffled_labels", lambda y, rng: y)
    with pytest.raises(SystemExit, match="CONTROL FAILED"):
        run(mod, monkeypatch, *args)
    assert pd.read_csv(stage / "control.csv").val_accuracy.iloc[0] >= C.CONTROL_CEILING


@pytest.mark.parametrize("score,stops", [(0.2, False), (0.99, True)])
def test_cnn_row_reaches_the_gate(stage, monkeypatch, score, stops):
    monkeypatch.setattr(N, "train_cnn", lambda *a, **k: (None, [{"val_acc": score}], None, None))
    if stops:
        with pytest.raises(SystemExit, match="CONTROL FAILED"):
            run(N, monkeypatch, "--only", "cnn")
    else:
        run(N, monkeypatch, "--only", "cnn")


# The finding behind the design: clean simple models can sit far from 20%, and that must not stop a run.
@pytest.mark.parametrize("score", [0.0004, 0.478, 0.731])
def test_clean_scores_outside_the_band_do_not_stop(tmp_path, score):
    p = tmp_path / "control.csv"
    pd.DataFrame({"model": ["nb"], "val_accuracy": [score]}).to_csv(p, index=False)
    C.control_gate(p)


def test_missing_score_stops(tmp_path):
    p = tmp_path / "control.csv"
    pd.DataFrame({"model": ["cnn"], "val_accuracy": [np.nan]}).to_csv(p, index=False)
    with pytest.raises(SystemExit, match="CONTROL FAILED"):
        C.control_gate(p)


def test_old_failed_row_cannot_hide_behind_a_rerun(tmp_path):
    p = tmp_path / "control.csv"
    pd.DataFrame({"model": ["mlp", "cnn"], "val_accuracy": [0.95, 0.2]}).to_csv(p, index=False)
    N.merge_rows(p, pd.DataFrame({"model": ["cnn"], "val_accuracy": [0.21]}), "model")
    with pytest.raises(SystemExit, match="1 of 2 rows"):
        C.control_gate(p)


def test_fair_shuffle_passes_and_smoke_only_prints(monkeypatch, capsys):
    y = np.arange(5000) % 5
    monkeypatch.setattr(C, "SMOKE", False)
    assert (C.shuffled_labels(y, np.random.default_rng(0)) != y).any()
    monkeypatch.setattr(C, "SMOKE", True)
    C.shuffled_labels(y, NoShuffle())
    assert "no verdict" in capsys.readouterr().out
