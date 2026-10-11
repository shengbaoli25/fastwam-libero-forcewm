"""Actual four-rank B16 two-Adam probe; never substitutes tiny-model tests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]


def validate_config(cfg):
    expected={'batch_size':16,'gradient_accumulation_steps':1,'max_steps':30000,
        'save_every':5000,'learning_rate':1e-4,'weight_decay':.01,'mixed_precision':'bf16',
        'seed':42,'resume':None,'lr_scheduler_type':'cosine'}
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Exact B16/accum1/30000 original recipe required')
    m=cfg['model'];d=cfg['data']['train']
    if (m['_target_']!='fastwam.forcewm.runtime.create_fastwam_forcewm'
            or d['_target_']!='fastwam.forcewm.data.ForceWMCachedDataset'
            or not d.get('expected_identity_sha256') or not d.get('sidecar_root')
            or not d.get('require_production') or not m.get('require_pretrained_fastwam')
            or m['generated_video_steps']!=16 or m['lambda_force']!=1.
            or m['force_hidden_dim']!=1280 or m['force_ffn_dim']!=4608 or m['force_num_heads']!=16
            or m['video_dit_config']['num_layers']!=30 or m['action_dit_config']['num_layers']!=30):
        raise ValueError('Bound production ForceWM architecture/cache required')


def helpers():
    spec=importlib.util.spec_from_file_location('baseline_probe_helpers',ROOT/'scripts/forcevla/preflight_cached_gpu.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def execute(config_path,output):
    import torch
    from accelerate import Accelerator
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam import runtime
    from fastwam.trainer import Wan22Trainer
    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed
    from .integration import architecture_counts
    start=time.monotonic(); cfg=OmegaConf.load(config_path)
    validate_config(OmegaConf.to_container(cfg,resolve=True))
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA hardware required')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.cuda.reset_peak_memory_stats()
    accelerator=Accelerator(gradient_accumulation_steps=1,mixed_precision='bf16',step_scheduler_with_optimizer=False)
    if accelerator.num_processes!=4 or accelerator.state.deepspeed_plugin is None:
        raise RuntimeError('Actual four DeepSpeed ranks required')
    output=Path(output)
    if accelerator.is_main_process:output.mkdir(parents=True,exist_ok=False)
    accelerator.wait_for_everyone()
    rank=accelerator.process_index;rank_dir=output/f'rank_{rank:03d}';rank_dir.mkdir()
    cfg.output_dir=str(rank_dir);misc.register_work_dir(str(rank_dir));set_global_seed(42)
    cfgfile=OmegaConf.to_container(cfg,resolve=True)
    (rank_dir/'config.json').write_text(json.dumps(cfgfile,indent=2))
    model=instantiate(cfg.model,model_dtype=torch.bfloat16,device=runtime._resolve_train_device())
    def forbidden(*args,**kw):raise RuntimeError('Cached ForceWM training called VAE encode/decode')
    model._encode_video_latents=forbidden
    for module in (model.vae,getattr(model.vae,'model',None)):
        if module is not None:
            for name in ('encode','decode'):
                if hasattr(module,name):setattr(module,name,forbidden)
    train_ds,val_ds=runtime.build_datasets(cfg.data)
    if val_ds is not None:raise RuntimeError('Baseline has no invented validation split')
    dataset_receipt=train_ds.audit_integrity()
    trainer=Wan22Trainer(model,train_ds,None,cfg=cfg)
    engine=trainer.model
    zero=trainer.accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']
    if (zero['stage']!=1 or engine.train_micro_batch_size_per_gpu()!=16
            or engine.gradient_accumulation_steps()!=1 or engine.train_batch_size()!=64
            or any(zero.get(k,{}).get('device','none')!='none' for k in ('offload_optimizer','offload_param'))):
        raise RuntimeError('Actual engine differs from four-rank global64 original ZeRO1')
    h=helpers();adam=h.unwrap_optimizer(trainer.optimizer)
    if not isinstance(adam,torch.optim.AdamW):raise RuntimeError('Actual AdamW required')
    if h.inspect_scheduler(trainer.scheduler)!={'last_epoch':0,'warmup_milestones':[1500],'cosine_steps':28500}:
        raise RuntimeError('Complete 30k learning rate schedule required')
    layer_calls=[0]*30;handles=[]
    for i,block in enumerate(model.mot.forcewm.blocks):
        def count(module,args,out,i=i):layer_calls[i]+=1
        handles.append(block.register_forward_hook(count))
    torch.cuda.synchronize();startup=time.monotonic()-start
    losses,updates=h.run_two_updates(trainer,synchronize=torch.cuda.synchronize)
    for handle in handles:handle.remove()
    report={'rank':rank,'world_size':4,'batch_per_rank':16,'global_batch':64,
        'actual_engine_updates':int(engine.global_steps),'scheduler':h.inspect_scheduler(trainer.scheduler),
        **h.inspect_adam(adam),'losses':losses,'updates':updates,'teacher_layer_calls':layer_calls,
        'all_parameters_finite':all(h._finite(p) for p in model.mot.parameters()),
        'startup_seconds':startup,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'gpu':torch.cuda.get_device_name(),'architecture_counts':architecture_counts(model),
        'dataset':dataset_receipt,'official_initialization':model.pretrained_fastwam_metadata,
        'memory_rng_state':model.memory_generator.get_state().tolist(),
        'formal_training_started':False,'checkpoint_saved':False,'distributed_resume_verified':False}
    accepted=(report['actual_engine_updates']==2 and report['adam_state_steps']
        and all(x==2 for x in report['adam_state_steps']) and report['all_parameters_finite']
        and report['adam_master_and_moments_finite'] and min(layer_calls)>=2
        and all(not u['optimizer_step_was_skipped'] for u in updates))
    report['accepted']=bool(accepted)
    (rank_dir/'preflight.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    ranks=[None]*4;torch.distributed.all_gather_object(ranks,report)
    if accelerator.is_main_process:
        accepted=all(r['accepted'] for r in ranks) and all(r['memory_rng_state']==ranks[0]['memory_rng_state'] for r in ranks)
        warm=max(r['updates'][-1]['seconds'] for r in ranks)
        result={'status':'two_adam_verified' if accepted else 'failed','ranks':ranks,
            'warm_seconds_per_step':warm,'training_only_30000_seconds_estimate':warm*30000,
            'save_eval_budget_verified':False,'formal_training_started':False,
            'checkpoint_saved':False,'distributed_resume_verified':False}
        (output/'preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in result.items() if k!='ranks'}),flush=True)
    accelerator.wait_for_everyone()
    ok=json.loads((output/'preflight.json').read_text())['status']=='two_adam_verified'
    torch.distributed.destroy_process_group()
    return ok


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'plan_only','world_size':4,'B':16,'global_batch':64,'schedule':30000,'probe_updates':2}));return
    try:ok=execute(args.config,args.output)
    except Exception as e:
        import sys
        result={'status':'failed','error_type':type(e).__name__,'error':str(e),'rank':int(os.environ.get('RANK',0))}
        t=sys.modules.get('torch')
        if t is not None and t.cuda.is_initialized():result['peak_allocated_gib']=t.cuda.max_memory_allocated()/2**30
        print(json.dumps(result),flush=True)
        if Path(args.output).is_dir():(Path(args.output)/f'failure_rank_{result["rank"]}.json').write_text(json.dumps(result,indent=2))
        raise
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
