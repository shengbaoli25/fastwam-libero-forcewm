"""Autoregressive Force extension of the complete official Video30/Action30 policy.

No stock source file or baseline guard is changed. Disabled Force executes the
inherited official loss/core verbatim. Enabled Force uses generated Video only,
then refreshes its free autoregressive prediction under the current Action each call.
Teacher-forced Force states are isolated from the Action expert.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_dit import flash_attention
from .model import (ActionDualMemory, AutoregressiveForceWM, forward_joint_with_memories,
                    masked_force_mse, shifted_wrench_inputs)

SCHEMA = "fastwam-forcewm-causal-ar30-v1"


class ForceRuntimeState(nn.Module):
    """Persistent per-rank memory RNG and strict protocol bytes for ZeRO resume.

    Tensor buffers survive accelerator/DeepSpeed full-state save/load without a
    custom trainer hook or non-Tensor safetensors members. The reviewed ranks
    share seed44, batch16 and draw geometry; their states must remain identical.
    ZeRO1 saves one replicated model state, so this is not a rank-specific RNG
    persistence protocol. A different manifest is rejected before restoration.
    """
    def __init__(self, seed, manifest):
        super().__init__()
        self.register_buffer("rng_state", torch.Generator(device="cpu").manual_seed(seed).get_state())
        self.register_buffer("manifest_bytes", self._encode(manifest))
        self.register_load_state_dict_pre_hook(self._validate_restore)

    @staticmethod
    def _encode(manifest):
        data = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":")).encode()
        return torch.tensor(list(data), dtype=torch.uint8)

    def bind_manifest(self, manifest):
        self.manifest_bytes = self._encode(manifest).to(self.manifest_bytes.device)

    def _validate_restore(self, module, state_dict, prefix, *args):
        incoming = state_dict.get(prefix + "manifest_bytes")
        rng = state_dict.get(prefix + "rng_state")
        if incoming is None or not torch.equal(incoming.cpu(), self.manifest_bytes.cpu()):
            raise ValueError("Full-state Force protocol/source/initialization manifest differs")
        if not isinstance(rng, Tensor) or rng.dtype != torch.uint8 or rng.shape != self.rng_state.shape:
            raise ValueError("Full-state Force memory RNG is missing or malformed")
        # A fresh CPU generator validates the actual state, not merely its shape.
        torch.Generator(device="cpu").set_state(rng.cpu())

    def get_state(self):
        return self.rng_state.detach().cpu().clone()

    def set_state(self, state):
        torch.Generator(device="cpu").set_state(state.cpu())
        if state.shape != self.rng_state.shape:
            raise ValueError("Memory RNG shape differs")
        self.rng_state.copy_(state.to(self.rng_state.device))
        return self

    def manual_seed(self, seed):
        return self.set_state(torch.Generator(device="cpu").manual_seed(seed).get_state())

    def normal(self, shape):
        generator = torch.Generator(device="cpu").set_state(self.get_state())
        result = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
        self.set_state(generator.get_state())
        return result


@dataclass(frozen=True)
class GeneratedVideoMemory:
    latents: Tensor
    layers: tuple[Tensor, ...]
    is_pad: Tensor


def _action_config(action):
    return {"hidden_dim": action.hidden_dim, "action_dim": action.action_dim,
            "ffn_dim": action.ffn_dim, "text_dim": action.text_embedding[0].in_features,
            "freq_dim": action.freq_dim, "eps": action.blocks[0].norm1.eps,
            "num_heads": action.num_heads, "attn_head_dim": action.attn_head_dim,
            "num_layers": len(action.blocks), "use_gradient_checkpointing": False}


def architecture_counts(model):
    count=lambda m:sum(p.numel() for p in m.parameters())
    return {"video_parameters":count(model.video_expert),"action_parameters":count(model.action_expert),
        "force_parameters":count(model.mot.forcewm),"action_memory_parameters":count(model.mot.action_dual_memory),
        "optimizer_parameters":count(model.mot)+count(model.proprio_encoder),
        "force_teacher_passes":1,"force_rollout_steps":model.mot.forcewm.horizon,
        "generated_video30_forwards":model.generated_video_steps+1,
        "counts_are_memory_fit_evidence":False}


def attach_forcewm(model, *, force_init_seed=43, memory_seed=44, generated_video_steps=16,
                   lambda_force=1., checkpoint_force_layers=True, force_hidden_dim=1280,
                   force_ffn_dim=4608, force_num_heads=16, memory_attention_width=512,
                   memory_heads=8, allow_tiny_test_model=False):
    if hasattr(model.mot,"forcewm"):
        raise ValueError("Force WM already registered")
    if not allow_tiny_test_model and (len(model.video_expert.blocks)!=30 or len(model.action_expert.blocks)!=30
            or model.video_expert.hidden_dim!=3072 or model.action_expert.hidden_dim!=1024):
        raise ValueError("Complete official Video30/Action30 geometry required")
    if model.video_expert.action_conditioned:
        raise ValueError("Baseline Video remains action-unconditioned")
    if type(generated_video_steps) is not int or generated_video_steps<1:
        raise ValueError("Positive generated video steps required")
    if not torch.isfinite(torch.tensor(lambda_force)) or lambda_force<=0:
        raise ValueError("Positive finite Force loss weight required")
    cfg=_action_config(model.action_expert)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(force_init_seed)
        force=AutoregressiveForceWM(video_hidden_dim=model.video_expert.hidden_dim,
            text_dim=model.text_dim,hidden_dim=force_hidden_dim,ffn_dim=force_ffn_dim,
            num_heads=force_num_heads,num_layers=model.mot.num_layers,
            checkpoint_layers=checkpoint_force_layers)
        memories=nn.ModuleList([ActionDualMemory(cfg,model.video_expert.hidden_dim,force_hidden_dim,
            memory_attention_width,memory_heads) for _ in range(model.mot.num_layers)])
    model.mot.add_module("forcewm",force.to(model.device,model.torch_dtype))
    model.mot.add_module("action_dual_memory",memories.to(model.device,model.torch_dtype))
    model.__class__=ForceWMFastWAM
    model.force_enabled=True;model.lambda_force=float(lambda_force)
    model.generated_video_steps=generated_video_steps
    model.forcewm_manifest={"schema":SCHEMA,"force_init_seed":force_init_seed,"memory_seed":memory_seed,
        "generated_video_steps":generated_video_steps,"lambda_force":float(lambda_force),
        "objective":"shifted teacher-forcing normalized6D MSE; inference deterministic AR from observed w0",
        "force_architecture":{"depth":model.mot.num_layers,"width":force_hidden_dim,"ffn":force_ffn_dim,
                              "heads":force_num_heads,"memory_width":memory_attention_width,"memory_heads":memory_heads},
        "Uc":"current noisy Action + current Action clock; no reference/candidate proposal",
        "Force_refresh":"free AR every Action denoise forward, final generated A0 at clock0",
        "teacher_inputs":"[w0,w1,...,w31] predict [w1,...,w32]; teacher states NEVER feed Action",
        "mask":"H causal H-only; U bidirectional U-only; W H+U+causalW; only W reads generated same-layer Video",
        "memory_gradient":"Video no-grad; Action reads detached free-rollout Force; teacher loss trains Force only",
        "official_initialization":copy.deepcopy(getattr(model,"pretrained_fastwam_metadata",None)),
        "force_initialization":"independent random, seed43; no baseline continuation",
        "video_semantics":"original observation-conditioned future; does not read Action",
        "history_offsets":list(range(-31,1)),"action_offsets":list(range(32)),"force_offsets":list(range(1,33)),
        "memory_rank_contract":"same seed44/batch16/forwardcount across ranks; replicated RNG state"}
    model.forcewm_manifest["architecture_counts"]=architecture_counts(model)
    model.memory_generator=ForceRuntimeState(memory_seed,model.forcewm_manifest)
    return model


class ForceWMFastWAM(FastWAM):
    @torch.no_grad()
    def _video_only(self, latents, clock, context, context_mask, fuse_flag):
        prepared = self.video_expert.prepare(x=latents, timestep=clock, context=context,
            context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag)
        x, t, mod, text, text_mask, freqs, f, h, w, per_frame = prepared
        mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=x.shape[1], video_tokens_per_frame=per_frame, device=x.device)
        layers = []
        for block in self.video_expert.blocks:
            q, k, v, residual, gate, shift, scale, mlp_gate, _ = self.mot._build_expert_attention_io(
                expert=self.video_expert, block=block, x=x, freqs=freqs, t_mod=mod)
            mixed = flash_attention(q, k, v, self.mot.num_heads, ctx_mask=mask)
            x = self.mot._apply_expert_post_block_tensor(block=block, residual_x=residual,
                mixed_attn_out=mixed, gate_msa=gate, shift_mlp=shift, scale_mlp=scale,
                gate_mlp=mlp_gate, context=text, context_mask=text_mask)
            # Same-layer generated *future* features, excluding observed frame.
            layers.append(x[:, per_frame:].detach())
        return self.video_expert.post(x, t, f, h, w), tuple(layers)

    @torch.no_grad()
    def generate_video_memory(self, first_frame_latents: Tensor, latent_shape: tuple,
                              context: Tensor, context_mask: Tensor, fuse_flag=True):
        if len(latent_shape) != 5 or latent_shape[2] < 2 or first_frame_latents.shape[2] != 1:
            raise ValueError("Generated memory requires observed first frame and explicit future latent shape")
        if tuple(first_frame_latents.shape) != (*latent_shape[:2], 1, *latent_shape[3:]):
            raise ValueError("Generated memory first-frame shape differs")
        # Never accepts teacher future latents or clean Action. Its independent
        # CPU generator is saved/resumed and does not consume V/A training RNG.
        x = self.memory_generator.normal(latent_shape)
        x = x.to(device=self.device, dtype=self.torch_dtype)
        x[:, :, :1] = first_frame_latents.detach()
        schedule, deltas = self.infer_video_scheduler.build_inference_schedule(
            self.generated_video_steps, self.device, x.dtype)
        for clock, delta in zip(schedule, deltas):
            prediction, _ = self._video_only(x, clock.expand(x.shape[0]), context, context_mask, fuse_flag)
            x = self.infer_video_scheduler.step(prediction, delta, x)
            x[:, :, :1] = first_frame_latents.detach()
        _, layers = self._video_only(x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype),
                                    context, context_mask, fuse_flag)
        if not layers or layers[0].shape[1] < 1:
            raise ValueError("Generated Video contains no future memory")
        return GeneratedVideoMemory(x.detach(), layers,
            torch.zeros(layers[0].shape[:2], dtype=torch.bool, device=x.device))

    def _force_condition(self, sample, noisy_action, action_clock, context, context_mask, memory):
        def field(key,dtype):
            return sample[key].to(self.device,dtype=dtype,non_blocking=True)
        return dict(history=field("history",self.torch_dtype),noisy_action=noisy_action.detach(),
            action_clock=action_clock.detach(),context=context.detach(),context_mask=context_mask,
            history_is_pad=field("history_is_pad",torch.bool),
            history_channel_valid=field("history_channel_valid",torch.bool),
            candidate_is_pad=sample.get("action_is_pad",torch.zeros(noisy_action.shape[:2],device=self.device,dtype=torch.bool)).to(self.device),
            observation_latents=memory.latents[:,:,:1],video_features=memory.layers,video_is_pad=memory.is_pad)

    def _force_prediction(self, sample, noisy_action, action_clock, context, context_mask, memory):
        return self.mot.forcewm.rollout(**self._force_condition(sample,noisy_action,action_clock,context,context_mask,memory))

    def training_loss(self, sample, tiled=False):
        if not self.force_enabled:
            return super().training_loss(sample, tiled=tiled)
        if hasattr(self, "_force_training_context"):
            raise RuntimeError("Nested/concurrent Force training calls are unsupported")
        state = {"sample": sample}
        self._force_training_context = state
        try:
            official_loss, losses = super().training_loss(sample, tiled=tiled)
        finally:
            del self._force_training_context
        prediction = state["teacher_prediction"]
        force_loss = masked_force_mse(prediction.wrench, sample["wrench_future"],
                                     sample["force_is_pad"], sample["wrench_valid"])
        losses["loss_force"] = self.lambda_force * float(force_loss.detach())
        return official_loss + self.lambda_force * force_loss, losses

    def _joint_denoise_core(self, latents_video, latents_action, timestep_video, timestep_action,
                            context, context_mask, attention_mask, fuse_vae_embedding_in_latents,
                            action_condition=None):
        if not self.force_enabled:
            return super()._joint_denoise_core(latents_video, latents_action, timestep_video,
                timestep_action, context, context_mask, attention_mask,
                fuse_vae_embedding_in_latents, action_condition)
        state = getattr(self, "_force_training_context", None)
        if state is None:
            raise ValueError("Force joint denoising requires explicit Force history; use infer_forcewm")
        memory = self.generate_video_memory(latents_video[:, :, :1], tuple(latents_video.shape),
                                            context, context_mask, fuse_vae_embedding_in_latents)
        prediction = self._force_prediction(state["sample"], latents_action, timestep_action,
                                            context, context_mask, memory)
        condition=self._force_condition(state["sample"],latents_action,timestep_action,context,context_mask,memory)
        sample=state["sample"]
        previous=shifted_wrench_inputs(condition["history"],sample["wrench_future"].to(self.device,self.torch_dtype),
            sample["force_is_pad"].to(self.device),sample["wrench_valid"].to(self.device))
        state["teacher_prediction"]=self.mot.forcewm.forward_teacher(previous_wrench=previous,**condition)
        state["free_prediction"]=prediction
        v = self.video_expert.prepare(x=latents_video, timestep=timestep_video, context=context,
            context_mask=context_mask, action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents)
        a = self.action_expert.prepare(action_tokens=latents_action, timestep=timestep_action,
                                      context=context, context_mask=context_mask)
        out_v, out_a = forward_joint_with_memories(self.mot, video_features=memory.layers,
            force_prediction=prediction, video_is_pad=memory.is_pad,
            action_is_pad=state["sample"]["action_is_pad"].to(self.device, torch.bool),
            video_tokens=v[0], action_tokens=a[0], video_freqs=v[5], action_freqs=a[5],
            video_t_mod=v[2], action_t_mod=a[2], video_context=v[3], video_context_mask=v[4],
            action_context=a[3], action_context_mask=a[4], attention_mask=attention_mask)
        return self.video_expert.post(out_v, v[1], v[6], v[7], v[8]), self.action_expert.post(out_a)

    @torch.no_grad()
    def infer_forcewm(self, sample, *, num_inference_steps=16, seed=0):
        """Cached-observation evaluation: free Video, iterative Action, paired final Force.

        All Force input fields must be observation/history and horizon metadata.
        Clean cached Action/Force labels are not read by this method.
        """
        if not self.force_enabled:
            raise ValueError("Force inference is disabled")
        self.eval()
        # Only the current observation is accepted from a cached window. Strip
        # future values *before* inherited validation, or accept the observed
        # frame directly. Neither a future latent nor any target is required.
        first = sample.get("first_frame_latents")
        if first is None:
            first = sample["video_latents"][:, :, :1]
        if first.ndim != 5 or first.shape[1:3] != (48, 1):
            raise ValueError("Force inference needs observed first_frame_latents [B,48,1,H,W]")
        num_frames = int(sample.get("video_num_frames", 9))
        if num_frames != 9:
            raise ValueError("The reviewed ForceWM recipe has an explicit nine-frame Video geometry")
        batch, channels, _, height, width = first.shape
        future_count = (num_frames - 1) // int(self.vae.temporal_downsample_factor) + 1
        placeholder = torch.zeros((batch, channels, future_count, height, width),
                                  device=first.device, dtype=first.dtype)
        placeholder[:, :, :1] = first
        horizon = self.mot.forcewm.horizon
        input_sample = {"video_latents": placeholder, "video_num_frames": num_frames,
            "action": torch.zeros((batch, horizon, self.action_expert.action_dim),
                device=self.device, dtype=self.torch_dtype),
            "action_is_pad": sample.get("action_is_pad", torch.zeros(batch, horizon, dtype=torch.bool))}
        for key in ("context", "context_mask", "prompt"):
            if key in sample:
                input_sample[key] = sample[key]
        if self.proprio_encoder is not None:
            current = sample["proprio"]
            if current.ndim == 3:
                current = current[:, 0]
            if current.shape != (batch, self.proprio_dim):
                raise ValueError("Force inference proprio must be current [B,8] (or cached [B,T,8])")
            input_sample["proprio"] = current[:, None].expand(-1, horizon, -1)
        inputs = self.build_inputs(input_sample)
        memory = self.generate_video_memory(inputs["first_frame_latents"],
            tuple(inputs["input_latents"].shape), inputs["context"], inputs["context_mask"],
            inputs["fuse_vae_embedding_in_latents"])
        gen = torch.Generator(device="cpu").manual_seed(seed)
        action = torch.randn(input_sample["action"].shape, generator=gen).to(self.device, self.torch_dtype)
        # Preserve official Action->observed-first-frame MoT attention separately
        # from the new generated future dual-memory residuals.
        frame = inputs["first_frame_latents"]
        v = self.video_expert.prepare(x=frame, timestep=torch.zeros(action.shape[0], device=self.device,
            dtype=self.torch_dtype), context=inputs["context"], context_mask=inputs["context_mask"],
            action=None, fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"])
        mask = self._build_mot_attention_mask(v[0].shape[1], action.shape[1], v[9], self.device)
        clocks, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps, self.device, action.dtype)
        for clock, delta in zip(clocks, deltas):
            t = clock.expand(action.shape[0])
            force = self._force_prediction(sample, action, t, inputs["context"], inputs["context_mask"], memory)
            a = self.action_expert.prepare(action, t, inputs["context"], inputs["context_mask"])
            _, out = forward_joint_with_memories(self.mot, video_features=memory.layers,
                force_prediction=force, video_is_pad=memory.is_pad, action_is_pad=inputs["action_is_pad"],
                video_tokens=v[0], action_tokens=a[0], video_freqs=v[5], action_freqs=a[5],
                video_t_mod=v[2], action_t_mod=a[2], video_context=v[3], video_context_mask=v[4],
                action_context=a[3], action_context_mask=a[4], attention_mask=mask)
            action = self.infer_action_scheduler.step(self.action_expert.post(out), delta, action)
        final_force = self._force_prediction(sample, action, torch.zeros(action.shape[0], device=self.device,
            dtype=action.dtype), inputs["context"], inputs["context_mask"], memory)
        return {"action": action.detach(), "wrench": final_force.wrench.detach(),
                "video_latents": memory.latents, "forcewm_manifest": copy.deepcopy(self.forcewm_manifest)}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {"forcewm_schema": SCHEMA, "forcewm_manifest": self.forcewm_manifest,
            "mot": self.mot.state_dict(), "step": step, "torch_dtype": str(self.torch_dtype),
            "memory_generator_state": self.memory_generator.get_state()}
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("forcewm_schema") != SCHEMA or payload.get("forcewm_manifest") != self.forcewm_manifest:
            raise ValueError("Force checkpoint architecture/initialization/conditioning manifest differs")
        if "mot" not in payload or "memory_generator_state" not in payload:
            raise ValueError("Complete V/A/F/memory and RNG checkpoint required")
        self.mot.load_state_dict(payload["mot"], strict=True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        elif "proprio_encoder" in payload:
            raise ValueError("Unexpected proprio checkpoint")
        self.memory_generator.set_state(payload["memory_generator_state"])
        if optimizer is not None:
            if "optimizer" not in payload:
                raise ValueError("Optimizer checkpoint required for resumed training")
            optimizer.load_state_dict(payload["optimizer"])
        return payload
