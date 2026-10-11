"""Strict official initialization for the separate ForceWM training recipe."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

from fastwam.baseline.runtime import create_fastwam_baseline
from .integration import attach_forcewm

OFFICIAL_RELEASE_SHA = "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"


def forcewm_source_inventory():
    root = Path(__file__).parent
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.glob("*.py"))}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()}


def create_fastwam_forcewm(*, force_init_seed=43, memory_seed=44, generated_video_steps=16,
                          lambda_force=1., checkpoint_force_layers=True,
                          force_hidden_dim=1280, force_ffn_dim=4608, force_num_heads=16,
                          memory_attention_width=512, memory_heads=8,
                          pretrained_fastwam_sha256=OFFICIAL_RELEASE_SHA,
                          require_pretrained_fastwam=True, **kwargs):
    if pretrained_fastwam_sha256 != OFFICIAL_RELEASE_SHA or require_pretrained_fastwam is not True:
        raise ValueError("ForceWM must initialize V/A from the same pinned complete official LIBERO release")
    if kwargs.get("compile_training_denoise", False):
        raise ValueError("ForceWM's separate generated-memory path has no reviewed compile protocol")
    model = create_fastwam_baseline(pretrained_fastwam_sha256=OFFICIAL_RELEASE_SHA,
        require_pretrained_fastwam=True, **kwargs)
    model = attach_forcewm(model, force_init_seed=force_init_seed, memory_seed=memory_seed,
        generated_video_steps=generated_video_steps, lambda_force=lambda_force,
        checkpoint_force_layers=checkpoint_force_layers,
        force_hidden_dim=force_hidden_dim, force_ffn_dim=force_ffn_dim, force_num_heads=force_num_heads,
        memory_attention_width=memory_attention_width, memory_heads=memory_heads)
    model.forcewm_manifest["source_inventory"] = forcewm_source_inventory()
    model.memory_generator.bind_manifest(model.forcewm_manifest)
    return model
