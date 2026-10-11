"""Run the inherited trainer from an exact resolved ForceWM config; record ranks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--origins',required=True)
    args=parser.parse_args()
    import torch
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from fastwam import runtime
    from fastwam.forcewm.preflight import validate_config
    from fastwam.forcewm.runtime import forcewm_source_inventory
    from fastwam.utils.pytorch_utils import set_global_seed
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    accelerator=Accelerator(gradient_accumulation_steps=1,mixed_precision='bf16',step_scheduler_with_optimizer=False)
    if accelerator.num_processes!=4 or rank!=accelerator.process_index or accelerator.state.deepspeed_plugin is None:
        raise RuntimeError('Formal ForceWM needs actual ranks0..3')
    cfg=OmegaConf.load(args.config);validate_config(OmegaConf.to_container(cfg,resolve=True))
    set_global_seed(42)
    Path(args.origins).mkdir(parents=True,exist_ok=True)
    file=Path(args.origins)/f'rank_{rank:03d}.json'
    receipt={'rank':rank,'world_size':4,'batch_per_rank':16,'global_batch':64,'max_steps':30000,
             'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
             'python_invocation':sys.executable,'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
             'forcewm_source_inventory':forcewm_source_inventory(),'output':str(cfg.output_dir)}
    with file.open('x') as stream:json.dump(receipt,stream,indent=2)
    try:
        runtime.run_training(cfg)
        state=Path(cfg.output_dir)/'checkpoints/state/step_030000/trainer_state.json'
        if json.loads(state.read_text())['global_step']!=30000:
            raise RuntimeError('Trainer exited without the complete30000 state')
        receipt['status']='complete'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error));raise
    finally:
        receipt['finished_at']=datetime.now(timezone.utc).isoformat()
        file.write_text(json.dumps(receipt,indent=2))
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
