"""CPU regression for the actual stock dataset factory used by the GPU probe."""
import pytest
from omegaconf import OmegaConf
from fastwam import runtime
from fastwam.forcewm.preflight import build_probe_datasets


def test_actual_stock_no_validation_split_preserves_dataset_alias():
    cfg = OmegaConf.create({'train': {'_target_': 'builtins.list'}})
    train, val = build_probe_datasets(cfg)
    assert train is val
    assert isinstance(train, list)


def test_explicit_validation_split_is_rejected():
    cfg = OmegaConf.create({'train': {'_target_': 'builtins.list'},
                           'val': {'_target_': 'builtins.list'}})
    with pytest.raises(ValueError, match='no separate validation'):
        build_probe_datasets(cfg)


def test_factory_contract_drift_cannot_silently_pass(monkeypatch):
    monkeypatch.setattr(runtime, 'build_datasets', lambda cfg: ([], []))
    with pytest.raises(RuntimeError, match='must alias'):
        build_probe_datasets(OmegaConf.create({'train': {}}))
