"""Resume a stopped four-rank ForceWM run in a foreground platform training job.

Read the old interactive checkpoint without changing it. The resumed job writes
to a fresh output; submit only after the old allocation is confirmed stopped.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from checks import (complete_state, force_deadline, launch_environment,
                    platform_identity, stopped_evidence, verify_restored, verify_code,
                    prior_resource, final_check, storage_estimate, budget_estimate)


def recipe(cfg):
    from omegaconf import OmegaConf
    from fastwam.forcewm.preflight import validate_config

    data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    data.pop("output_dir", None)
    data["resume"] = None
    validate_config(data)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_inputs(source: Path, prior: Path, output: Path, prior_log: Path):
    from omegaconf import OmegaConf
    from fastwam.forcewm.runtime import forcewm_source_inventory

    if not source.is_dir() or not prior.is_dir() or output.exists():
        raise ValueError("Source/prior output missing or new output already exists")
    config = prior / "config.yaml"
    cfg = OmegaConf.load(config)
    digest = recipe(cfg)
    if Path(cfg.output_dir) != prior:
        raise ValueError("Prior config belongs to another output")
    manifest_path = prior / "resume_origin.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    root = Path(manifest["root_output"]) if manifest else prior
    if root != Path("/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/fastwam-forcewm-ab5d6993-795b-4bf7-9084-e59f089e7788-4rank-30k-v9"):
        raise ValueError("Resume lineage must bind the actual original experiment")
    root_config = root / "config.yaml"
    if (not root_config.is_file() or recipe(OmegaConf.load(root_config)) != digest
            or (manifest and (manifest.get("output") != str(prior)
                              or manifest.get("root_config_sha256") != hashlib.sha256(root_config.read_bytes()).hexdigest()
                              or manifest.get("recipe_sha256") != digest))):
        raise ValueError("Original training recipe/source chain changed")
    expected = forcewm_source_inventory()
    if manifest and manifest.get("source_inventory") != expected:
        raise ValueError("ForceWM model code changed across resume jobs")
    origins = prior / "origins" if manifest else Path(str(prior) + "_origins")
    for rank in range(4):
        receipt = json.loads((origins / f"rank_{rank:03d}.json").read_text())
        if manifest:
            valid = (receipt.get("rank") == rank and receipt.get("job") == manifest.get("job")
                     and receipt.get("resume_step") == manifest.get("resume_step")
                     and receipt.get("status") in ("resumed", "complete"))
        else:
            valid = (receipt.get("rank") == rank and receipt.get("world_size") == 4
                     and receipt.get("batch_per_rank") == 16 and receipt.get("global_batch") == 64
                     and receipt.get("max_steps") == 30000 and receipt.get("output") == str(prior)
                     and receipt.get("forcewm_source_inventory") == expected
                     and receipt.get("status") in ("running", "complete", "failed"))
        if not valid:
            raise ValueError("Four matching source-rank receipts required")
    if manifest is None and cfg.resume is not None:
        raise ValueError("Original run must start from the official Base")
    if manifest is not None and str(cfg.resume) != manifest.get("resume_state"):
        raise ValueError("Prior resume configuration differs from its recorded lineage")
    state, step = complete_state(prior, prior_log)
    return config, state, step, root, digest


def worker(args):
    import torch
    from accelerate import Accelerator
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam import runtime
    from fastwam.trainer import Wan22Trainer
    from fastwam.utils import misc
    from fastwam.utils.logging_config import setup_logging
    from fastwam.utils.pytorch_utils import set_global_seed

    verify_code(args.source)
    cfg = OmegaConf.load(args.config)
    if recipe(cfg) != args.recipe_sha256 or Path(cfg.output_dir) != args.prior_output:
        raise ValueError("Checkpoint does not belong to this config")
    if complete_state(args.prior_output, args.prior_log) != (args.state, args.step):
        raise ValueError("Selected full-state checkpoint changed before rank startup")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    acc = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16",
                      step_scheduler_with_optimizer=False)
    if acc.num_processes != 4 or acc.state.deepspeed_plugin is None:
        raise ValueError("Resume requires exactly four DeepSpeed ranks")
    if acc.process_index != int(os.environ["RANK"]):
        raise ValueError("Rank identity mismatch")
    cfg.output_dir = str(args.output)
    cfg.resume = str(args.state)
    set_global_seed(42)
    if acc.is_main_process:
        OmegaConf.save(cfg, args.output / "config.yaml")
    acc.wait_for_everyone()
    origins = args.output / "origins"
    origins.mkdir(exist_ok=True)
    receipt = {"rank": acc.process_index, "status": "starting", "resume_step": args.step,
               "state": str(args.state), "job": os.environ["TRAIN_JOB_ID"],
               "started_at": datetime.now(timezone.utc).isoformat()}
    origin = origins / f"rank_{acc.process_index:03d}.json"
    with origin.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    setup_logging(is_main_process=acc.is_main_process)
    misc.register_work_dir(str(args.output))
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=runtime._resolve_train_device())
    train_ds, val_ds = runtime.build_datasets(cfg.data)
    trainer = Wan22Trainer(model, train_ds, val_ds, cfg=cfg)
    from continuation import restore_saved_native_rng
    native_rng_restore = restore_saved_native_rng(trainer, args.state)
    from fastwam.forcewm.preflight import helpers
    restored = verify_restored(trainer, args.state, args.step, helpers())
    restored["native_rng_restore"] = native_rng_restore
    engine = trainer.model
    if (engine.train_micro_batch_size_per_gpu(), engine.gradient_accumulation_steps(),
            engine.train_batch_size()) != (16, 1, 64):
        raise ValueError("Actual distributed batch differs from B16/global64")
    receipt.update(status="resumed", restored_step=trainer.global_step,
                   engine_step=engine.global_steps, restored=restored)
    origin.write_text(json.dumps(receipt, indent=2))
    from continuation import install_continuation
    install_continuation(trainer)
    acc.wait_for_everyone()
    trainer.train()
    if trainer.global_step != 30000:
        raise RuntimeError("Resume ended before 30000 steps")
    receipt.update(status="complete", global_step=30000)
    origin.write_text(json.dumps(receipt, indent=2))
    acc.wait_for_everyone()
    torch.distributed.destroy_process_group()


def controller(args):
    job = platform_identity(os.environ)
    termination = stopped_evidence(args.stop_evidence, args.prior_output)
    if termination["resource_id"] != prior_resource(args.prior_output, "ab5d6993-795b-4bf7-9084-e59f089e7788"):
        raise ValueError("Termination evidence belongs to another allocation")
    source = args.source.resolve()
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError("Entry point must belong to the selected source")
    frozen = verify_code(source)
    sys.path[:0] = [str(source / "src"), str(source), str(source / "scripts/forcewm")]
    config, state, step, root, digest = verify_inputs(source, args.prior_output, args.output, args.prior_log)
    deadline = force_deadline(args.budget_receipt, time.time())
    budget = budget_estimate(args.prior_log, step, deadline, time.time())
    native = args.native_python
    if (native != Path("/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python")
            or not native.is_absolute() or not native.is_file() or native.parent.name != "bin"
            or native.parent.parent.name != ".venv"):
        raise ValueError("Original absolute .venv/bin/python invocation required")
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               check=True, capture_output=True, text=True).stdout.splitlines()
    if len(inventory) not in (4, 8) or any("H200" not in name for name in inventory):
        raise ValueError("Allocate four or eight H200 GPUs; run exactly four training ranks")
    if args.output.exists() or args.watch.exists():
        raise FileExistsError("Resume output/watch must be fresh")
    storage = storage_estimate(state, args.output, step)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          check=True, capture_output=True, text=True).stdout
    if apps.strip():
        raise ValueError("Allocated GPUs already have compute processes")
    from bounded_probe import execute
    env = launch_environment(os.environ, source)
    command = [str(native), "-m", "accelerate.commands.launch", "--config_file",
               str(source / "scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml"),
               "--num_processes", "4", "--num_machines", "1", str(Path(__file__).resolve()),
               "--worker", "--source", str(source), "--prior-output", str(args.prior_output),
               "--prior-log", str(args.prior_log),
               "--output", str(args.output), "--state", str(state), "--step", str(step),
               "--config", str(config), "--recipe-sha256", digest]
    plan = {"budget": budget, "frozen_code_commit": frozen["commit"], "storage": storage, "prior_output": str(args.prior_output), "job": job, "source": str(source), "source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "resume_state": str(state), "resume_step": step, "output": str(args.output),
            "root_output": str(root), "root_config_sha256": hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest(),
            "absolute_deadline_unix": deadline, "budget_receipt_sha256": hashlib.sha256(args.budget_receipt.read_bytes()).hexdigest(),
            "stop_evidence_sha256": hashlib.sha256(args.stop_evidence.read_bytes()).hexdigest(),
            "recipe_sha256": digest, "source_inventory": __import__("fastwam.forcewm.runtime", fromlist=["forcewm_source_inventory"]).forcewm_source_inventory(),
            "argv": command, "gpu_names": inventory, "started_at": datetime.now(timezone.utc).isoformat()}
    print("RESUME_PLAN", json.dumps(plan), flush=True)
    if not args.execute:
        return
    args.watch.mkdir(parents=True, exist_ok=False)
    (args.watch / "resume-plan.json").write_text(json.dumps(plan, indent=2))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "resume_origin.json").write_text(json.dumps(plan, indent=2))
    os.environ.clear()
    os.environ.update(env)
    seconds = min(172800, deadline - time.time() - 60)
    result = execute(command, str(source), str(args.watch / "supervision"), seconds,
                     maximum_seconds=172800, deadline_unix=deadline,
                     log_name="training.log", phase="forcewm-distributed-resume")
    print("RESUME_SUPERVISOR", json.dumps(result), flush=True)
    if result["status"] != "process_exited_zero":
        raise RuntimeError("Distributed resume did not exit cleanly")
    completion = final_check(args.output, args.watch / "supervision/training.log", job)
    (args.watch / "completion.json").write_text(json.dumps(completion, indent=2))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--execute", action="store_true")
    for name in ("source", "prior-output", "prior-log", "stop-evidence", "output", "watch", "native-python", "state", "config", "recipe-sha256", "budget-receipt"):
        parser.add_argument("--" + name)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    for name in ("source", "prior_output", "prior_log", "stop_evidence", "output", "watch", "native_python", "state", "config", "budget_receipt"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value))
    if args.worker:
        worker(args)
    else:
        if any(getattr(args, name) is None for name in
               ("source", "prior_output", "prior_log", "stop_evidence", "output", "watch", "native_python", "budget_receipt")):
            parser.error("Controller requires source, prior-output, prior-log, stop-evidence, output, watch, native-python, and budget-receipt")
        controller(args)


if __name__ == "__main__":
    main()
