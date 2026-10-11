# FastWAM LIBERO dual-camera + autoregressive Force WM

This experiment inherits `shengbaoli25/fastwam-libero` commit
`a417e69cb3ed3c8ee9d56fe150e42d89a0547468`. Video30, Action30, MoT, the
dual-camera cache, and the optimizer schedule are retained. Video and Action
initialize from the **complete official LIBERO Base**, not a trained baseline
checkpoint. Force WM initializes independently with seed43.

The user-supplied PDF, *视频世界模型选择｜Cosmos-Predict2.5-2B*, page3,
SHA256 `9ec3a7f757ac34d54671a22122880dcb362f436d2973b0c2d8600c1cf23cada8`,
is the reference for stream isolation and same-layer visual cross-attention.
This implementation retains FastWAM's video backbone and 30-layer Action
expert. It does not substitute Cosmos28/AE14, add a GPT planner, or require
reference/candidate trajectories. The user explicitly requested autoregressive
supervision: the PDF's bidirectional W queries are replaced by causal,
right-shifted continuous wrench tokens.

## Force prediction and masks

The numeric adapter reconstructs measured FP64 episode-local streams:
history `H=[state,wrench]` at offsets `[-31..0]`, Action at `[0..31]`, and
future wrench targets at `[1..32]`. Normalization uses the baseline's pinned
statistics. Padding is determined before clamping indices; measured values
are retained separately for evaluation. Thirty denotes **layers**, not horizon.

Training inputs to W are `[w0,w1,...,w31]`, supervised against `[w1,...,w32]`.
Inference starts from observed `w0`; each six-axis output is the next input.
Force has no Gaussian initialization, diffusion clock, denoising objective,
or force flow matching. Its MSE averages measured axes/tokens per sample.
An Action clock remains an input condition describing the current Action
sample; it is not a Force diffusion clock.

Rows below are queries; columns are key/value streams:

| Force-internal Q → KV | H | U | W |
|---|---|---|---|
| H | causal | blocked | blocked |
| U | blocked | bidirectional | blocked |
| W | all observed H | all current U | causal (including shifted self) |

H additionally receives pooled observed-frame VAE latent and masked text.
U is a separate encoding of current `A_sigma` and its Action clock; it never
reads Action-expert hidden states. Only W cross-attends to same-layer free
Video features. H/U cannot acquire W or generated future video at later layers.
U is bidirectional, so this is causal in **force outputs**, conditional on the
whole current action plan; it does not claim strict action-prefix causality.

| Module Q → KV | Video | Force | Action |
|---|---|---|---|
| Video | original FastWAM mask | blocked | blocked |
| Force | W-only cross-attention | mask above | blocked (raw U separately encoded) |
| Action | original observed-video attention + free-video memory | free AR rollout only | original bidirectional Action |

Teacher-forced Force states **never feed Action**. Training makes a separate,
detached free rollout for Action memory, preventing future-force labels from
leaking across branches. It also computes one teacher-forced Force pass for
the MSE. Inference refreshes free Force at every Action forward and once more
with the final generated Action at clock0.

## Video and caching

Reuse the complete immutable 34808-window VAE feature cache: main left,
wrist right, each224×224, concatenated224×448. Each feature was independently
encoded from nine frames `0:33:4`; never slice episode-level temporal latents.
Training reads cached latent/action/proprio/text and a small numeric sidecar.
There is no RGB decoding or VAE encode/decode in the training path. The probe
replaces VAE entrypoints with errors to detect accidental use.

Force visual memory comes from Video's free prediction using observed frame
and original context. No teacher-noised future video reaches Force. This costs
16 generated-video steps plus a final same-layer feature pass per batch, even
though VAE features are cached. Force rollout caches per-layer H/U, previous-W
and projected Video K/V, so it does not recompute full prefixes32 times.
Video remains action-unconditioned: these outputs are not evidence of action
consequences, closed-loop control, or generalization.

## Hyperparameters

| Setting | Value |
|---|---|
| GPUs / processes | 4 H100 / 4 ranks |
| Per-GPU batch / accumulation / global batch | 16 / 1 / 64 |
| New training steps / checkpoint interval | 30000 / 5000 |
| Video / Action / Force depth | 30 / 30 / 30 |
| Force width / heads / FFN | 1280 / 16 / 4608 |
| Force parameters | 892596486 |
| New Action dual-memory parameters | 196761660 |
| Total trainable parameters | 7110068745 |
| Action memory attention width / heads | 512 / 8; zero residual gates initially |
| Force objective / weight | normalized six-axis MSE / 1.0 |
| Video / Action objective | original Gaussian flow matching / original weights1 |
| AdamW | lr1e-4, betas(0.9,0.95), weight_decay0.01 |
| Schedule | warmup1500, cosine28500 |
| Precision / sharding | BF16 / original ZeRO1, no optimizer offload |
| Seeds | baseline42 / Force43 / independent Video-memory RNG44 |
| Activation recomputation | Force blocks and new Action memory enabled |
| Full training resume | null; starts from official Base |

The width/heads/FFN follow the PDF; depth is the requested30. The actual
parameter count is0.893B, not the PDF's approximate0.7B. The original DS JSON
has no `gradient_clipping` key; trainer `max_grad_norm=1` does not establish
engine clipping at1. Source/config defaults are not proof of H100 fit.

## Running

Use a fresh source/output directory on the shared volume and the existing
project `.venv/bin/python` invocation. Keep the baseline source and cache
immutable. Resolve the Force task with actual asset and cache paths, prepare
`fastwam.forcewm.data.prepare_forcewm_sidecar(...)` in a separate directory,
and bind `data.train.expected_identity_sha256` and `sidecar_root` explicitly.

The distributed probe reuses the original trainer and full30000-step scheduler
but executes exactly two disposable Adam updates. It measures real B16/global64,
finite losses/weights/Adam states, step2, all30 Force layers, no skipped update,
memory, time and cache provenance. It saves no model and is not a formal launch.

```bash
PYTHONPATH="$PWD/src:$PWD" /path/to/original/.venv/bin/python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml \
  --module fastwam.forcewm.preflight --config /path/to/resolved.yaml \
  --output /path/to/fresh-probe --execute
```

For bounded execution, put that exact argv list in JSON and use
`scripts/forcewm/bounded_probe.py --execute --command-json ... --cwd ...
--output ... --seconds 1800` under stdlib Python with Linux pidfd support.
The copied reviewed process tracker records PID/start_ticks and independent
sessions, cleans owned descendants on failure/deadline, and restores subreaper.
The supervisor's rc0 only proves process exit; separately require the actual
four-rank `two_adam_verified` receipt and unforced closure.

Only after measured fit and an actual full-run time/storage budget, launch
`scripts/train.py task=forcevla_libero_2cam_cached_forcewm_4gpu_30k` with the same
resolved overrides and four-rank launcher. Do not shorten batch/steps, reduce
Video generation steps, or change optimizer silently to pass the probe.

## Validation boundaries

CPU operator tests cover causal leakage, shifted labels, all30 layers,
cached-vs-full-prefix AR equivalence, baseline disabled parity, no future GT
in inference, teacher isolation with open Action gates, gradient checkpointing,
B16 synthetic training, immutable numeric sidecars and weights/RNG save/load.
CPU results do not prove GPU fit, distributed restore, real force accuracy,
or30000-step completion. Wrench source units/calibration remain unverified.
Real evaluation must report free-rollout errors, not teacher-forced MSE alone.
