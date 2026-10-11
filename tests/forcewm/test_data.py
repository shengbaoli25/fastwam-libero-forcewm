"""CPU synthetic fixtures for same-cache Force history/target adaptation."""
from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from fastwam.cache.latent import (LATENT_SHAPE, CacheIntegrityError, LatentCacheDataset,
                                 cache_identity_sha256, file_sha256, make_cache_identity,
                                 write_window_cache)
from fastwam.datasets.lerobot.utils import normalizer as stock_normalizer
from fastwam.datasets.lerobot.utils.normalizer import (SingleFieldLinearNormalizer,
                                                      load_dataset_stats_from_json)
from fastwam.forcewm import data as force_data
from fastwam.forcewm.data import ForceWMCachedDataset, prepare_forcewm_sidecar


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _bits_equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.reshape(-1).contiguous().view(torch.uint8), right.reshape(-1).contiguous().view(torch.uint8))


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=True) + "\n")


def _fixture(root, *, stats_change=None, source_pin=None):
    root.mkdir(parents=True)
    lengths = [3, 4]
    stats_path = root / "stats.json"
    stats = {"state": {"default": {"global_mean": [.12, -.03, .5, .04, .1, -.08, .7],
                                    "global_std": [.25, .5, .08, .75, 1., .9, .2]}},
             "action": {"default": {"global_mean": [1., 2., 3., 4., 5., 6., 7.],
                                     "global_std": [2., 3., 4., 5., 6., 7., 8.]}},
             "dualwm_force": {"global_mean": [2.1, -1.2, 0., .35, 1e-6, -.05],
                               "global_std": [2.4, 3.2, .7, .12, 1e-7, .025]},
             "num_episodes": 2, "num_transition": 7}
    if stats_change:
        stats_change(stats)
    _write_json(stats_path, stats)
    loaded = load_dataset_stats_from_json(stats_path)
    norms = {kind: SingleFieldLinearNormalizer(
        {"mean": loaded[kind]["default"]["global_mean"], "std": loaded[kind]["default"]["global_std"]},
        mode="z-score") for kind in ("state", "action")}
    state = torch.tensor([[index / 3 + axis / 13 for axis in range(7)] for index in range(7)], dtype=torch.float64)
    wrench = torch.tensor([[index / 5 + axis / 29 for axis in range(6)] for index in range(7)], dtype=torch.float64)
    state[0, 0] = wrench[0, 0] = -0.0
    state[1, 0] = wrench[1, 0] = 0.0
    state[2, 1] = wrench[2, 1] = 0.12345678901234567
    raw_action = state + 5.3
    normalized_state, normalized_action = norms["state"].forward(state.float()), norms["action"].forward(raw_action.float())
    proprio = torch.cat((normalized_state, torch.zeros(7, 1)), dim=-1)
    prompt = "TEST-only Force task"
    identity = make_cache_identity(data_conversion_sha256=_hash("TEST-only conversion"),
        dataset_stats_sha256=file_sha256(stats_path), vae_weights_sha256=_hash("TEST-only fake VAE"),
        text_encoder_weights_sha256=_hash("TEST-only fake T5"), preprocessing={"TEST-only": True},
        source_code_sha256={force_data.NORMALIZER_SOURCE: source_pin or file_sha256(stock_normalizer.__file__)},
        text_payload_sha256={prompt: _hash(prompt)}, episode_lengths=lengths, test_fixture=True,
        encoding_backend="TEST-only fakeencoder")
    rows = [{"sample_index": index, "episode_index": 0 if index < 3 else 1,
             "frame_index": index if index < 3 else index - 3, "task_index": 0} for index in range(7)]
    calls = []
    def getter(index):
        row = rows[index]
        frame, episode = row["frame_index"], row["episode_index"]
        start, length = index - frame, lengths[episode]
        raw_frames = frame + torch.arange(32)
        indices = start + raw_frames.clamp(max=length - 1)
        return {"sample_index": index, "video": torch.full((3, 9, 224, 448), float(index)),
                "action": normalized_action[indices], "proprio": proprio[indices],
                "image_is_pad": frame + torch.arange(0, 33, 4) >= length,
                "action_is_pad": raw_frames >= length,
                "proprio_is_pad": frame + torch.arange(33) >= length,
                "raw_current_state": state[index], "raw_current_wrench": wrench[index],
                "raw_action": raw_action[indices], "raw_wrench_future": wrench[indices], "prompt": prompt}
    def fake_encoder(video):
        calls.append(float(video[0, 0, 0, 0, 0]))
        return torch.full((1, *LATENT_SHAPE), calls[-1], dtype=torch.bfloat16)
    cache = root / "cache"
    manifest = write_window_cache(cache, identity, getter, rows, fake_encoder,
        {prompt: {"context": torch.zeros(128, 4096, dtype=torch.bfloat16),
                  "context_mask": torch.ones(128, dtype=torch.bool)}}, shard_size=2)
    return {"cache": cache, "stats": stats_path, "identity": manifest["identity_sha256"],
            "state": state, "wrench": wrench, "manifest": manifest, "encoder_calls": calls}


def _dataset(fixture, **kwargs):
    return ForceWMCachedDataset(fixture["cache"], fixture["stats"], fixture["identity"],
                                require_production=False, **kwargs)


@pytest.fixture
def source(tmp_path):
    return _fixture(tmp_path / "source")


@pytest.mark.parametrize("index", range(7))
def test_every_episode_boundary_anchor_history_and_plus_one_target_are_raw_bitwise_exact(source, index):
    dataset = _dataset(source)
    sample = dataset[index]
    episode_start, episode_length = (0, 3) if index < 3 else (3, 4)
    frame = index - episode_start
    h_indices = [episode_start + max(0, min(frame + offset, episode_length - 1)) for offset in range(-31, 1)]
    f_indices = [episode_start + min(frame + offset, episode_length - 1) for offset in range(1, 33)]
    expected_h_pad = torch.tensor([frame + offset < 0 for offset in range(-31, 1)])
    expected_f_pad = torch.tensor([frame + offset >= episode_length for offset in range(1, 33)])
    assert _bits_equal(sample["raw_history_proprio"], source["state"][h_indices])
    assert _bits_equal(sample["raw_wrench_history"], source["wrench"][h_indices])
    assert _bits_equal(sample["raw_wrench_future"], source["wrench"][f_indices])
    assert _bits_equal(sample["raw_wrench_future64"], source["wrench"][f_indices])
    assert torch.equal(sample["history_is_pad"], expected_h_pad)
    assert torch.equal(sample["force_is_pad"], expected_f_pad)
    assert torch.equal(sample["wrench_valid"], (~expected_f_pad[:, None]).expand(32, 6))
    assert torch.equal(sample["history_channel_valid"], (~expected_h_pad[:, None]).expand(32, 13))
    assert sample["history"].shape == (32, 13)
    assert sample["history_proprio"].shape == (32, 7)  # No padded eighth proprio dimension.
    assert len(source["encoder_calls"]) == 7  # Dataset never calls any encoder.


def test_stock_action_latents_proprio_text_preserved_and_old_raw_force_kept_for_audit(source):
    dataset = _dataset(source)
    stock = LatentCacheDataset(source["cache"], source["identity"], require_production=False)
    for index in range(7):
        before, after = stock[index], dataset[index]
        for key in ("video_latents", "action", "proprio", "image_is_pad", "action_is_pad", "proprio_is_pad",
                    "context", "context_mask", "raw_current_state", "raw_current_wrench", "raw_action"):
            assert _bits_equal(before[key], after[key])
        assert _bits_equal(before["raw_wrench_future"], after["raw_wrench_future_stock"])
        assert before["prompt"] == after["prompt"]
    first = dataset[0]
    assert _bits_equal(first["raw_wrench_future_stock"][0], source["wrench"][0])
    assert _bits_equal(first["raw_wrench_future"][0], source["wrench"][1])
    assert first["action_is_pad"].sum() + 1 == first["force_is_pad"].sum()


def test_normalized_inputs_use_stock_state_stats_and_independent_force_stats_and_padding_zero(source):
    dataset = _dataset(source)
    stats = load_dataset_stats_from_json(source["stats"])
    state_norm = SingleFieldLinearNormalizer({"mean": stats["state"]["default"]["global_mean"],
        "std": stats["state"]["default"]["global_std"]}, mode="z-score")
    force_norm = SingleFieldLinearNormalizer({"mean": stats["dualwm_force"]["global_mean"],
        "std": stats["dualwm_force"]["global_std"]}, mode="z-score")
    for index in range(7):
        sample = dataset[index]
        h_pad, f_pad = sample["history_is_pad"], sample["force_is_pad"]
        state = state_norm.forward(sample["raw_history_proprio"].float()).masked_fill(h_pad[:, None], 0)
        history_force = force_norm.forward(sample["raw_wrench_history"].float()).masked_fill(h_pad[:, None], 0)
        force = force_norm.forward(sample["raw_wrench_future"].float()).masked_fill(f_pad[:, None], 0)
        assert _bits_equal(sample["history_proprio"], state)
        assert _bits_equal(sample["wrench_history"], history_force)
        assert _bits_equal(sample["history"], torch.cat((state, history_force), dim=-1))
        assert _bits_equal(sample["wrench_future"], force)
        assert sample["history"].dtype == sample["wrench_future"].dtype == torch.float32
        assert sample["raw_wrench_future"].dtype == torch.float64
        assert bool((sample["wrench_future"].abs() <= 5).all())
    assert dataset[6]["force_is_pad"].all() and not dataset[6]["wrench_valid"].any()
    assert torch.count_nonzero(dataset[6]["wrench_future"]) == 0
    # Raw truth preserves both signed zero and more precision than the model input.
    assert torch.signbit(dataset[0]["raw_wrench_history"][-1, 0])
    assert source["wrench"][2, 1] != source["wrench"][2, 1].float().double()


def test_samples_are_copies_and_adapter_preserves_rng_and_collates_heterogeneous_padding(source):
    rng = torch.get_rng_state().clone()
    dataset = _dataset(source)
    sample = dataset[0]
    original = dataset[0]
    for value in sample.values():
        if isinstance(value, torch.Tensor):
            value.zero_()
    again = dataset[0]
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            assert _bits_equal(value, again[key])
    batch = torch.utils.data.default_collate([dataset[0], dataset[6]])
    assert batch["history"].shape == (2, 32, 13)
    assert batch["wrench_future"].shape == (2, 32, 6)
    assert not torch.equal(batch["history_is_pad"][0], batch["history_is_pad"][1])
    assert torch.equal(torch.get_rng_state(), rng)


def test_external_sidecar_binds_full_source_and_reloads_without_changing_cache(source, tmp_path):
    manifest_bytes = (source["cache"] / "manifest.json").read_bytes()
    output = tmp_path / "force-sidecar"
    receipt = prepare_forcewm_sidecar(source["cache"], source["stats"], output, source["identity"], require_production=False)
    dataset = _dataset(source, sidecar_root=output)
    assert dataset.sidecar_identity_sha256 == receipt["identity_sha256"]
    assert receipt["binding"]["cache_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert receipt["binding"]["history_offsets"] == list(range(-31, 1))
    assert receipt["binding"]["force_offsets"] == list(range(1, 33))
    assert len(receipt["binding"]["cache_shards"]) == 4
    assert receipt["binding"]["dataset_stats_sha256"] == file_sha256(source["stats"])
    assert dataset.audit_integrity()["verified"] is True
    assert (source["cache"] / "manifest.json").read_bytes() == manifest_bytes
    plain = _dataset(source)
    for index in range(7):
        assert _bits_equal(plain[index]["history"], dataset[index]["history"])
        assert _bits_equal(plain[index]["raw_wrench_future"], dataset[index]["raw_wrench_future"])
    with pytest.raises(FileExistsError):
        prepare_forcewm_sidecar(source["cache"], source["stats"], output, source["identity"], require_production=False)


@pytest.mark.parametrize("relative", [".", "child"])
def test_sidecar_cannot_overlap_immutable_cache(source, relative):
    with pytest.raises(CacheIntegrityError, match="external"):
        prepare_forcewm_sidecar(source["cache"], source["stats"], source["cache"] / relative,
                               source["identity"], require_production=False)


def test_sidecar_cannot_enclose_cache_or_overwrite_stats(source):
    with pytest.raises(CacheIntegrityError, match="external"):
        prepare_forcewm_sidecar(source["cache"], source["stats"], source["cache"].parent,
                               source["identity"], require_production=False)


def test_production_requires_full_production_cache_and_exact_expected_identity(source):
    with pytest.raises(CacheIntegrityError, match="TEST-only"):
        ForceWMCachedDataset(source["cache"], source["stats"], source["identity"])
    with pytest.raises(CacheIntegrityError, match="identity"):
        _dataset({**source, "identity": _hash("wrong")})


def _rewrite_shard(source, change, *, shard_number=0):
    path = source["cache"] / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["shards"][shard_number]
    payload_path = source["cache"] / entry["path"]
    payload = load_file(str(payload_path))
    change(payload)
    save_file(payload, str(payload_path))
    entry["sha256"] = file_sha256(payload_path)
    entry["tensor_shapes"] = {key: list(value.shape) for key, value in payload.items()}
    _write_json(path, manifest)


def test_complete_six_measured_channels_required_missing_force_cannot_be_zero_filled(source):
    _rewrite_shard(source, lambda payload: payload.pop("raw_current_wrench"))
    with pytest.raises(CacheIntegrityError, match="complete measured"):
        _dataset(source)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_force_channels_rejected_even_after_manifest_hash_is_updated(source, bad):
    _rewrite_shard(source, lambda payload: payload["raw_current_wrench"].__setitem__((0, 5), bad))
    with pytest.raises(CacheIntegrityError, match="nonfinite|finite"):
        _dataset(source)


def test_raw_state_must_agree_bitwise_with_pinned_state_normalization(source):
    _rewrite_shard(source, lambda payload: payload["raw_current_state"].__setitem__((0, 0), 0.01))
    with pytest.raises(CacheIntegrityError, match="normalization"):
        _dataset(source)


@pytest.mark.parametrize("change", [lambda stats: stats.pop("dualwm_force"),
    lambda stats: stats["dualwm_force"].update(global_std=[0.] * 6),
    lambda stats: stats["dualwm_force"].update(global_mean=[float("nan")] * 6),
    lambda stats: stats["dualwm_force"].update(global_std=[1.] * 5)])
def test_missing_nonfinite_wrong_width_or_nonpositive_force_stats_rejected(tmp_path, change):
    source = _fixture(tmp_path / "source", stats_change=change)
    with pytest.raises(CacheIntegrityError, match="Stats lack|stats require"):
        _dataset(source)


def test_stats_hash_and_stock_normalizer_source_pins_rejected_on_drift(source, tmp_path):
    source["stats"].write_text(source["stats"].read_text() + " ")
    with pytest.raises(CacheIntegrityError, match="stats SHA256"):
        _dataset(source)
    other = _fixture(tmp_path / "other", source_pin=_hash("different stock normalizer"))
    with pytest.raises(CacheIntegrityError, match="normalization source"):
        _dataset(other)


def test_exact_manifest_bytes_not_only_canonical_identity_guard_each_read(source):
    dataset = _dataset(source)
    dataset[0]  # Warm the stock shard cache first.
    path = source["cache"] / "manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=1))
    with pytest.raises(CacheIntegrityError, match="manifest bytes"):
        dataset[0]


def test_each_read_rejects_stats_and_adapter_source_drift(source, monkeypatch):
    dataset = _dataset(source)
    hashes = copy.deepcopy(dataset.binding["source_code_sha256"])
    hashes["adapter"] = _hash("adapter source drift")
    monkeypatch.setattr(force_data, "_source_hashes", lambda: hashes)
    with pytest.raises(CacheIntegrityError, match="source changed"):
        dataset[0]
    monkeypatch.undo()
    source["stats"].write_text(source["stats"].read_text() + " ")
    with pytest.raises(CacheIntegrityError, match="stats changed"):
        dataset[0]


def test_history_future_source_shard_drift_is_caught_after_cached_reads_even_with_restored_mtime(source):
    dataset = _dataset(source)
    dataset[0]
    entry = source["manifest"]["shards"][1]  # Next raw rows, used by anchor0's +1 targets.
    path = source["cache"] / entry["path"]
    stat = path.stat()
    payload = load_file(str(path))
    payload["raw_current_wrench"][0, 0] += .1
    save_file(payload, str(path))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(CacheIntegrityError, match="shard changed"):
        dataset[0]


def test_cached_text_drift_rejected_after_text_is_loaded(source):
    dataset = _dataset(source)
    path = source["cache"] / source["manifest"]["texts"][0]["path"]
    payload = load_file(str(path))
    payload["context"][0, 0] = 1
    save_file(payload, str(path))
    with pytest.raises(CacheIntegrityError, match="cached text changed"):
        dataset[0]


def test_stale_sidecar_rejected_when_valid_cache_manifest_bytes_change(source, tmp_path):
    output = tmp_path / "sidecar"
    prepare_forcewm_sidecar(source["cache"], source["stats"], output, source["identity"], require_production=False)
    path = source["cache"] / "manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text()), indent=1))
    with pytest.raises(CacheIntegrityError, match="bindings differ"):
        _dataset(source, sidecar_root=output)


def test_rehashed_forged_sidecar_raw_signed_zero_is_rejected_against_cache(source, tmp_path):
    output = tmp_path / "sidecar"
    receipt = prepare_forcewm_sidecar(source["cache"], source["stats"], output, source["identity"], require_production=False)
    path = output / "numeric.safetensors"
    payload = load_file(str(path))
    payload["raw_wrench"][0, 0] = 0.0  # Numerically equal but different original bytes.
    save_file(payload, str(path))
    receipt["numeric"]["sha256"] = file_sha256(path)
    receipt["identity_sha256"] = cache_identity_sha256({key: value for key, value in receipt.items() if key != "identity_sha256"})
    _write_json(output / "manifest.json", receipt)
    with pytest.raises(CacheIntegrityError, match="raw values differ"):
        _dataset(source, sidecar_root=output)


@pytest.mark.parametrize("target", ["manifest", "numeric"])
def test_sidecar_drift_after_construction_rejected(source, tmp_path, target):
    output = tmp_path / "sidecar"
    prepare_forcewm_sidecar(source["cache"], source["stats"], output, source["identity"], require_production=False)
    dataset = _dataset(source, sidecar_root=output)
    if target == "manifest":
        path = output / "manifest.json"
        path.write_text(path.read_text() + " ")
    else:
        path = output / "numeric.safetensors"
        payload = load_file(str(path))
        payload["raw_wrench"][0, 1] += .1
        save_file(payload, str(path))
    with pytest.raises(CacheIntegrityError, match="sidecar .* changed"):
        dataset[0]


@pytest.mark.parametrize("index", [-1, 7, True, 1.0])
def test_invalid_anchor_never_changes_sample(source, index):
    with pytest.raises(IndexError):
        _dataset(source)[index]
