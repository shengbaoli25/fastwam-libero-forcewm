"""CPU replay contracts; no production CUDA restore claim."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/resume"))
from continuation import (restore_saved_native_rng, rng_snapshot, restore_rng,
                          verify_rng_and_sampler)


@pytest.mark.parametrize("rank", range(4))
def test_final_restore_replays_saved_draws_and_keeps_strict_gate(tmp_path, monkeypatch, rank):
    initial = rng_snapshot()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        target = rng_snapshot()
        saved = dict(random_state=target[0], numpy_random_seed=target[1],
                     torch_manual_seed=target[2], step=10000)
        torch.save(saved, tmp_path / f"random_states_{rank}.pkl")
        expected = (random.random(), np.random.random(), torch.rand(5))
        # All native generators and accumulation step can be changed by load
        # bookkeeping. Reapply the saved rank state, never a new seed.
        random.seed(7)
        np.random.seed(8)
        torch.manual_seed(9)
        model = torch.nn.Module()
        model.register_buffer("rng_state", torch.Generator().manual_seed(44).get_state())
        model.register_buffer("manifest_bytes", torch.tensor([1, 2], dtype=torch.uint8))
        (tmp_path / "pytorch_model").mkdir()
        torch.save({"module": model.state_dict()}, tmp_path / "pytorch_model/mp_rank_00_model_states.pt")
        acc = SimpleNamespace(process_index=rank, step=0, unwrap_model=lambda _: model)
        trainer = SimpleNamespace(accelerator=acc, model=model, train_loader=list(range(544)),
            train_sampler=SimpleNamespace(epoch_offset=18, resume_batch_offset=208,
                                          seed=42, batch_size=16, num_processes=4))
        report = restore_saved_native_rng(trainer, tmp_path)
        assert report["source"] == f"random_states_{rank}.pkl"
        assert not report["before_matches"]["torch_cpu"]
        assert acc.step == 10000
        assert verify_rng_and_sampler(trainer, tmp_path, {"epoch": 18, "batch_in_epoch": 208})["python_numpy_torch_cuda"]
        actual = (random.random(), np.random.random(), torch.rand(5))
        assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
        restore_saved_native_rng(trainer, tmp_path)
        model.rng_state.copy_(torch.Generator().manual_seed(999).get_state())
        with pytest.raises(ValueError, match="Force memory"):
            verify_rng_and_sampler(trainer, tmp_path, {"epoch": 18, "batch_in_epoch": 208})
    finally:
        restore_rng(initial)


@pytest.mark.parametrize("damage", ["cpu_dtype", "cpu_shape", "cuda_count", "step", "python", "numpy", "missing"])
def test_invalid_saved_rank_state_is_refused(tmp_path, monkeypatch, damage):
    initial = rng_snapshot()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        py, npy, cpu, _ = rng_snapshot()
        saved = dict(random_state=py, numpy_random_seed=npy, torch_manual_seed=cpu, step=10000)
        if damage == "cpu_dtype": saved["torch_manual_seed"] = cpu.float()
        elif damage == "cpu_shape": saved["torch_manual_seed"] = cpu[:1]
        elif damage == "cuda_count": saved["torch_cuda_manual_seed"] = [cpu]
        elif damage == "step": saved["step"] = -1
        elif damage == "python": saved["random_state"] = None
        elif damage == "numpy": saved["numpy_random_seed"] = None
        elif damage == "missing": del saved["torch_manual_seed"]
        torch.save(saved, tmp_path / "random_states_0.pkl")
        trainer = SimpleNamespace(accelerator=SimpleNamespace(process_index=0, step=0))
        with pytest.raises((ValueError, TypeError, KeyError)):
            restore_saved_native_rng(trainer, tmp_path)
        after = rng_snapshot()
        assert after[0] == py and np.array_equal(after[1][1], npy[1]) and torch.equal(after[2], cpu)
    finally:
        restore_rng(initial)
