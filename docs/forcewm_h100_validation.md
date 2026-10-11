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
