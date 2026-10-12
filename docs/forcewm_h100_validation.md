# H100 ForceWM AR30 validation

Observed on2026-10-12, Shanghai time, allocation
`b2c145b8-a54c-46e0-8580-919ec0549792` (h100), nodeqb-prod-gpu128.
Four NVIDIA H10080GB HBM3 cards each report81559MiB, Torch2.7.1+cu128.

## Actual GPU probe

Source `b9cfc5ca32879d012aa446630e2085e64019b433`; the subsequent formal
launch guards do not change the seven validated ForceWM library files.
Probe v2 ran00:41:38–00:45:06. All four actual ranks usedB16, accumulation1,
global64 and the original full30000-step schedule. Each completed two Adam
updates, all Adam state steps2, no skipped updates, finite model/master/moment
and loss tensors, and all30 Force layers executed. VAE entrypoints were
replaced by errors during the probe, confirming training used cached features.

| Actual measurement | Value |
|---|---:|
| Total trainable parameters | 7,110,068,745 |
| Peak allocated / maximum reserved per GPU | 76.6214 / 76.9180GiB |
| Worst startup | 169.9432s |
| Slowest warm second update | 3.829035s |
| Training-only30000-step estimate | 31.9086h |
| Update1 total/video/action/force loss | 8.20534 /0.45249 /6.03971 /1.71315 |
| Update2 total/video/action/force loss | 6.75035 /0.44618 /4.80940 /1.49477 |

The supervisor exited0 after204 tracked processes in5 owned groups closed.
No cleanup signals, ownership errors or remaining owned processes; subreaper
was restored. EvidenceZIP113481bytes/35members, SHA256
`6e435f092a82168bc912a61d37d5772407caa3d897881223bf7aa995fd619122`;
CRC and member size/SHA checks passed locally. The earlier v1 probe failed
before Adam due to an overly strict no-split dataset check; the stock factory
returns the training object as validation. That check was fixed and three
factory regressions passed. V1 failure evidence remains preserved.

## Formal budget and limits

The user approved40h total from allocation creation at2026-10-12 00:02:25
Shanghai: the unchanged deadline is2026-10-13 16:02:25
(2026-10-13T08:02:25Z, Unix1791878545).
The formal recipe remains4ranks/B16/accum1/global64/30000/save5000,
BF16/original ZeRO1/lr1e-4/warmup1500/cosine/weight_decay0.01.
V/A start from complete official Base, Force seed43, memory RNG44.

Two updates are not a sustained32-hour speed measurement. Native saving has
six unique snapshots and seven save calls (periodic and final30000 both save).
Approximately0.683TB for six complete snapshots is an estimate; a successful
full checkpoint save and fresh four-rank restore have not yet been established.
No full30000 completion or actual evaluation is claimed by this document.
Wrench units/calibration and held-out generalization remain unverified.

CPU validation:60 original operator/data/process cases passed, then3 new
stock dataset-factory regressions. The formal guard adds18 focused cases;
three existing descendant/process tests were also rerun after its common
supervisor changed. CPU fixtures do not establish GPU training or model results.

## Oct12 formal failure and supervisor correction

The formal run did not complete. The final observed native log is Oct12
02:24:41 Shanghai, step1280/30000 (4.2667%), total/action/video/force loss
0.2440/0.0783/0.0857/0.0800, 0.27 steps/s. This is the last logged step,
not proof that no unlogged updates occurred before termination.

At02:24:57 the supervisor's system-wide `/proc` scan raised
`Invalid Linux process identity` for PID66897, stateZ, PPID0, PGID/session−1,
start_ticks1906410121. That PID was not in its tracked training tree. The
exception path initiated owned-tree SIGTERM and then SIGKILL cleanup. The
native launcher received signal15; the final supervisor receipt is failed,
returncode−9 at02:25:02.696981, with closure verified and subreaper restored.
This is forced failure cleanup, not a successful training completion.

The platform subsequently stopped the idle allocation at05:46:49 for manager
CPU/GPU/MEM utilization auto-recycle rules, finishing the image save/stop
at05:47:24. It did not reach the user-authorized40h deadline. No CUDA OOM or
nonfinite-loss exception appears in the retrieved training log.

Normal platform Project Files listings show both checkpoint weights and state
directories empty. The run had not reached its first5000-step checkpoint.
The saved container image preserves the environment, not the lost in-memory
training state; step1280 cannot be resumed from the available files.

Evidence SHA256 values:

- Actual38032-byte training log: `704a4495fd0b4e46770d09a21b08c0d7edcfc15d95043a9fdf7d0c4867eea9ba`.
- Actual260679-byte supervisor receipt: `f17e507c46f4b2925014ae5c67e4d489c86e1f4463d5beae5f54d8349a5e3780`.
- Actual132-byte completion receipt: `b02e958c0432b3aff38679ceb7ffd9e961b217859e01c9b4257e9bb72ae07735`.
- Actual2083-byte platform event log: `4d6bb3cf6cf705afe99fc61b99ab3618aa6e2ba3a9b3416aac6989a01e98bcbb`.

The correction extends the existing X-state disappearance check to Z-state
exit races. A second `/proc` read must actually report the PID gone. A persistent
invalid row, reused PID, or denied read remains an error; valid zombies stay
tracked for reaping. No PID is skipped merely for being a zombie. Signals still
require pidfd and matching PID/start_ticks, with the same full-tree cleanup and
deadline enforcement. Model code, cached features and training recipe do not
change.

The incident's first invalid sample is captured; its immediate second sample
was not captured by the old code, so the exact kernel timing cannot be replayed
from production evidence. The disappearing-Z scenario reproduces the failure
in CPU tests. Before the change, the original12 regression cases had3 failures
(including a real four-session supervisor run) and9 passes. After the change,
14 focused exit-race cases plus the existing21 process/formal guard cases
passed together:35 passed in4.52s. These are CPU/process results, not a new GPU
training run or a full save/restore proof.

The failed source/output/receipts are retained unchanged. No container restart
or new formal run occurred during this diagnosis. Around08:54 Shanghai,
approximately31.1h remains before the original fixed deadline, below the
existing36.19h start/save/eval budget estimate for a fresh30000-step run.
Any later run needs an explicitly approved budget and a fresh bound plan/output;
the old40h deadline must not be reset silently.
