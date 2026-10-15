"""Synthetic CPU monitoring checks; no patient datasets or performance claims."""

import io
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from survot_rank.training import train_runner as runner


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def args(tmp_path, method="dct_v22", **overrides):
    values = dict(survot_method=method, bag_loss="nll_surv", alpha_surv=0.,
                  cur_fold=0, cur_epoch=0, batch_size=4, max_epochs=1,
                  max_smoke_batches=0, grad_accum_steps=1, grad_clip_norm=0.,
                  rna_format="RNASeq", num_workers=0, results_dir=str(tmp_path),
                  min_free_space_gb=0., evaluation_protocol="legacy_val")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_patient_weighting_respects_missing_modalities_and_unequal_batches():
    monitoring = runner._PatientWeightedMonitoring()
    monitoring.add({"wsi_monitor_patient_count": 4, "wsi_slot_pair_dist_mean": 2,
                    "omics_monitor_patient_count": 1, "omics_slot_pair_dist_mean": 8,
                    "transport_monitor_patient_count": 1, "transport_mass_mean": .2,
                    "optimization_main_nll": 2}, 4)
    monitoring.add({"wsi_monitor_patient_count": 1, "wsi_slot_pair_dist_mean": 7,
                    "omics_monitor_patient_count": 0, "omics_slot_pair_dist_mean": 99,
                    "transport_monitor_patient_count": 0, "transport_mass_mean": 99,
                    "optimization_main_nll": 7}, 1)
    result = monitoring.means()
    assert result["wsi_slot_pair_dist_mean"] == pytest.approx(3)
    assert result["omics_slot_pair_dist_mean"] == 8
    assert result["transport_mass_mean"] == .2
    assert result["optimization_main_nll"] == pytest.approx(3)
    assert result["wsi_monitor_patient_count"] == 5
    assert result["omics_monitor_patient_count"] == 1
    assert result["transport_monitor_patient_count"] == 1


def test_unavailable_invalid_and_training_only_observations_are_excluded():
    monitoring = runner._PatientWeightedMonitoring()
    monitoring.add({"wsi_slot_pair_dist_mean": 9, "bad": float("nan"),
                    "vector": torch.ones(2), "global_nll": 7, "auxiliary_total": 8,
                    "omics_slot_norm_mean": 3}, 4,
                   excluded={"global_nll", "auxiliary_total"},
                   metric_weights={"omics_slot_norm_mean": 2})
    assert monitoring.means() == {"omics_slot_norm_mean": 3}
    monitoring.add({"omics_slot_norm_mean": 99}, 4,
                   metric_weights={"omics_slot_norm_mean": 0})
    assert monitoring.means() == {"omics_slot_norm_mean": 3}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.child = torch.nn.Dropout(.5)
        self.last_training_losses = {}
        self.last_monitoring_metrics = {"stale": 999}
        self.last_monitoring_metric_weights = {"stale": 999}
        self.last_evaluation_monitoring_metrics = {"stale": 999}
        self.monitoring_before_forward = []


class SyntheticLoader(list):
    def __init__(self, items, batch_size=1):
        super().__init__(items)
        self.batch_size = batch_size
        self.sampler = torch.utils.data.SequentialSampler(range(len(items)))
        self.drop_last = False
        self.dataset = SimpleNamespace(label_df=pd.DataFrame({
            "case id": [f"case{i}" for i in range(len(items) * batch_size)],
        }))


class SizedDataset:
    def __init__(self, frame):
        self.label_df = frame

    def __len__(self):
        return len(self.label_df)


def install_synthetic_forward(monkeypatch):
    def process(config, model, data, device, test=False):
        model.monitoring_before_forward.append(dict(model.last_monitoring_metrics))
        batch_size = data.get("n", 1)
        if "monitor" in data:
            model.last_monitoring_metrics = data["monitor"]
            model.last_monitoring_metric_weights = data.get("weights", {})
        model.last_training_losses = {"reference_deficit": torch.tensor(data.get("deficit", .1))}
        model.last_training_losses.update({
            name: torch.tensor(value) for name, value in data.get("training_losses", {}).items()
        })
        logits = (model.parameter + float(data["nll"])).expand(batch_size, 2)
        auxiliary = model.parameter.square() + float(data.get("aux", 0))
        return ((logits, auxiliary), torch.zeros(batch_size, dtype=torch.long),
                torch.arange(batch_size, dtype=torch.float32) + 1, torch.zeros(batch_size))
    monkeypatch.setattr(runner, "_process_data_and_forward", process)
    monkeypatch.setattr(runner, "_calculate_metrics", lambda *a, **kw: (.7, .7, .2, .2, .7))
    monkeypatch.setattr(runner, "concordance_index_censored", lambda *a, **kw: (.6,))


def simple_sum_loss(logits, *labels):
    return logits[:, 0].sum()


def test_training_logs_primary_nll_separately_from_objective_and_clears_stale(monkeypatch, tmp_path):
    install_synthetic_forward(monkeypatch)
    model = TinyModel()
    loader = [{"n": 4, "nll": 2, "aux": 3, "deficit": .9,
               "monitor": {"wsi_monitor_patient_count": 4, "wsi_slot_pair_dist_mean": 2}},
              {"n": 1, "nll": 6, "aux": 9, "deficit": .1}]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    result = runner.train_one_epoch(args(tmp_path), 0, model, loader, optimizer,
                                   SimpleNamespace(step=lambda: None), simple_sum_loss, io.StringIO())
    assert result["optimization_main_nll"] == pytest.approx(2.8)
    assert result["optimization_auxiliary_total"] == pytest.approx(4.2)
    assert result["optimization_total_objective"] == pytest.approx(7)
    assert result["reference_deficit"] == pytest.approx(.74)
    assert result["wsi_monitor_patient_count"] == 4
    assert result["wsi_slot_pair_dist_mean"] == 2
    assert result["optimization_patient_count"] == 5
    assert result["optimization_cindex"] == .6
    assert model.monitoring_before_forward == [{}, {}]


def test_old_method_training_keeps_its_schema(monkeypatch, tmp_path):
    install_synthetic_forward(monkeypatch)
    model = TinyModel()
    result = runner.train_one_epoch(args(tmp_path, "dct_v21"), 0, model,
                                   [{"n": 4, "nll": 2, "deficit": .9},
                                    {"n": 1, "nll": 6, "deficit": .1}],
                                   torch.optim.SGD(model.parameters(), lr=0.),
                                   SimpleNamespace(step=lambda: None), simple_sum_loss, io.StringIO())
    assert result == {"reference_deficit": pytest.approx(.5)}
    assert model.monitoring_before_forward[0] == {"stale": 999}


def test_sampled_relation_gradient_uses_samples_once_without_batch_size_dilution(monkeypatch, tmp_path):
    install_synthetic_forward(monkeypatch)
    model = TinyModel()
    def sampled(size, norm):
        metrics = {"relation_cost_gradient_sample_count": 1,
                   "relation_cost_gradient_norm_raw": norm}
        return {"n": size, "nll": 2, "monitor": metrics,
                "training_losses": dict(metrics),
                "weights": {name: 1 for name in metrics}}
    loader = [sampled(4, 3),
              {"n": 2, "nll": 2, "monitor": {"relation_cost_gradient_sample_count": 0},
               "weights": {"relation_cost_gradient_sample_count": 1}},
              sampled(1, 9)]
    result = runner.train_one_epoch(args(tmp_path), 0, model, loader,
                                   torch.optim.SGD(model.parameters(), lr=0.),
                                   SimpleNamespace(step=lambda: None), simple_sum_loss, io.StringIO())
    assert result["relation_cost_gradient_sample_count"] == 2
    assert result["relation_cost_gradient_norm_raw"] == pytest.approx(6)
    assert result["optimization_patient_count"] == 7


def test_evaluation_normalizes_processed_patients_and_keeps_ids_after_missing_case(monkeypatch, tmp_path):
    install_synthetic_forward(monkeypatch)
    model = TinyModel()
    loader = SyntheticLoader([
        {"nll": 2, "monitor": {"global_nll": 999, "auxiliary_total": 999,
                               "wsi_monitor_patient_count": 1, "wsi_slot_norm_mean": 3}},
        None, {"nll": 6}, {"nll": 4}])
    loader.dataset = SizedDataset(loader.dataset.label_df)
    result = runner.evaluate(args(tmp_path), None, model, loader, simple_sum_loss)
    assert result[-1] == pytest.approx(4)
    assert set(result[0]) == {"case0", "case2", "case3"}
    assert model.last_evaluation_monitoring_metrics == {
        "wsi_monitor_patient_count": 1, "wsi_slot_norm_mean": 3,
        "main_nll": 4, "patient_count": 3,
    }
    assert model.monitoring_before_forward == [{}, {}, {}]
    runner.evaluate(args(tmp_path), None, model, SyntheticLoaderWithSize([{"nll": 1}]), simple_sum_loss)
    assert model.last_evaluation_monitoring_metrics == {"main_nll": 1, "patient_count": 1}


class SyntheticLoaderWithSize(SyntheticLoader):
    def __init__(self, items):
        super().__init__(items)
        self.dataset = SizedDataset(self.dataset.label_df)


def test_empty_evaluation_does_not_leave_stale_monitoring(monkeypatch, tmp_path):
    install_synthetic_forward(monkeypatch)
    model = TinyModel()
    with pytest.raises(ValueError, match="processed no patients"):
        runner.evaluate(args(tmp_path), None, model, SyntheticLoaderWithSize([None]), simple_sum_loss)
    assert model.last_evaluation_monitoring_metrics == {}


@pytest.mark.parametrize("policy", ["batched", "replacement", "unordered", "drop_last"])
def test_v22_rejects_evaluation_policies_that_cannot_preserve_patient_ids(monkeypatch, tmp_path, policy):
    def forbidden(*a, **kw):
        raise AssertionError("invalid loader must fail before any model forward")
    monkeypatch.setattr(runner, "_process_data_and_forward", forbidden)
    loader = SyntheticLoaderWithSize([{"nll": 2}, {"nll": 4}])
    if policy == "batched":
        loader.batch_size = 2
    elif policy == "replacement":
        loader.sampler = torch.utils.data.RandomSampler(range(2), replacement=True)
    elif policy == "unordered":
        loader.sampler = torch.utils.data.RandomSampler(range(2))
    else:
        loader.drop_last = True
    model = TinyModel()
    with pytest.raises(ValueError, match="sequential batch_size=1"):
        runner.evaluate(args(tmp_path), None, model, loader, simple_sum_loss)
    assert model.last_evaluation_monitoring_metrics == {}


def test_old_method_evaluation_keeps_historical_return_and_no_monitoring_reset(monkeypatch, tmp_path):
    install_synthetic_forward(monkeypatch)
    model = TinyModel()
    result = runner.evaluate(args(tmp_path, "dct_v21"), None, model,
                             SyntheticLoaderWithSize([{"nll": 2}, {"nll": 4}]), simple_sum_loss)
    assert len(result) == 7
    assert result[-1] == 3
    assert model.last_evaluation_monitoring_metrics == {"stale": 999}


@pytest.mark.parametrize("fails", [False, True])
def test_train_snapshot_restores_mixed_module_modes(monkeypatch, tmp_path, fails):
    model = TinyModel().train()
    model.child.eval()
    before = [module.training for module in model.modules()]
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    def evaluate(*a, **kw):
        model.eval()
        random.random()
        np.random.random()
        torch.rand(3)
        if fails:
            raise RuntimeError("synthetic failure")
        model.last_evaluation_monitoring_metrics = {"main_nll": 2.5}
        return ({}, .75, 0, 0, 0, 0, 2.5)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    if fails:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            runner._evaluate_v22_train_snapshot(args(tmp_path), None, model, None, None, None)
    else:
        assert runner._evaluate_v22_train_snapshot(args(tmp_path), None, model, None, None, None) == {
            "main_nll": 2.5, "cindex": .75,
        }
    assert [module.training for module in model.modules()] == before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_train_snapshot_loader_keeps_patients_and_original_sampling_state(monkeypatch, tmp_path):
    class FakeSurvivalDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.split_key = "inner_train"
            self.label_df = pd.DataFrame({"case id": ["trainA", "trainB"]})
        def __len__(self):
            return len(self.label_df)
        def __getitem__(self, index):
            return torch.tensor(index)
    monkeypatch.setattr(runner, "SurvivalDataset", FakeSurvivalDataset)
    dataset = FakeSurvivalDataset()
    original_labels = dataset.label_df.copy(deep=True)
    rng_before = torch.get_rng_state().clone()
    loader = runner._build_v22_train_evaluation_loader(args(tmp_path), dataset)
    assert [item.item() for item in loader] == [0, 1]
    assert loader.dataset is not dataset
    assert loader.dataset.split_key == "val"
    assert dataset.split_key == "inner_train"
    pd.testing.assert_frame_equal(dataset.label_df, original_labels)
    pd.testing.assert_frame_equal(loader.dataset.label_df, original_labels)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    assert loader.batch_size == 1 and loader.drop_last is False


@pytest.mark.parametrize("evaluate_train", [False, True])
def test_epoch_csv_keeps_validation_snapshot_when_train_evaluation_follows(monkeypatch, tmp_path, evaluate_train):
    model = TinyModel()
    labels = pd.DataFrame({"case id": ["A", "B"], "c": [0., 1.], "label": [0, 1], "time": [1., 2.]})
    dataset = SizedDataset(labels)
    val_loader = object()
    train_eval_loader = object()
    monkeypatch.setattr(runner, "get_split", lambda *a: (dataset, dataset, [], val_loader))
    monkeypatch.setattr(runner, "_build_v22_train_evaluation_loader", lambda *a: train_eval_loader)
    monkeypatch.setattr(runner, "init_model_for_method", lambda *a: model)
    monkeypatch.setattr(runner, "init_loss_function", lambda *a: simple_sum_loss)
    monkeypatch.setattr(runner, "init_optimizer", lambda *a: None)
    monkeypatch.setattr(runner, "init_scheduler", lambda *a: None)
    monkeypatch.setattr(runner, "_extract_survival_metadata", lambda *a: None)
    monkeypatch.setattr(runner, "train_one_epoch", lambda *a: {"optimization_main_nll": 1.2})
    def evaluate(*values):
        snapshot_loss = 3. if values[3] is val_loader else 1.
        model.last_evaluation_monitoring_metrics = {"main_nll": snapshot_loss, "transport_mass_mean": snapshot_loss / 10}
        return ({"A": {}}, .7, .7, .2, .2, .7, snapshot_loss)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    factory = SimpleNamespace(censorship_var="c", label_col="time")
    runner.train_one_fold(args(tmp_path, dct_v22_evaluate_train=evaluate_train), factory, 0, io.StringIO())
    curve = pd.read_csv(tmp_path / "epoch_curve_fold0.csv")
    assert curve.loc[0, "train_optimization_main_nll"] == 1.2
    assert curve.loc[0, "val_main_nll"] == 3
    assert curve.loc[0, "val_transport_mass_mean"] == pytest.approx(.3)
    if evaluate_train:
        assert curve.loc[0, "train_eval_main_nll"] == 1
        assert curve.loc[0, "train_eval_transport_mass_mean"] == pytest.approx(.1)
    else:
        assert not any(column.startswith("train_eval_") for column in curve)


@pytest.mark.parametrize("method", ["dct_v22", "dct_v2_2"])
def test_v22_uses_stable_sum_nll(method, tmp_path):
    loss = runner.init_loss_function(args(tmp_path, method))
    logits = torch.zeros(2, 2)
    value = loss(logits, torch.tensor([0, 1]), None, torch.tensor([0., 1.]))
    assert value.item() == pytest.approx(3 * np.log(2))
