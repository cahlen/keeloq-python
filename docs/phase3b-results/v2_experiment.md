# Phase 3b v2 Spatial-Conv Experiment

## Control: Δ search at depth 56 (v1 got best 0.688)

Wall clock: 172.1s — top 5:

| Δ | val_acc | loss |
|---|---:|---:|
| 0x00000002 | 0.7034 | 0.5961 |
| 0x00010000 | 0.6746 | 0.6034 |
| 0x00800000 | 0.6724 | 0.6048 |
| 0x00020000 | 0.6540 | 0.6262 |
| 0x02000000 | 0.6536 | 0.6222 |

## Primary: Δ search at depth 88 (v1 all < 0.517)

Wall clock: 259.6s — top 10:

| Δ | val_acc | loss |
|---|---:|---:|
| 0x00000020 | 0.5198 | 0.6932 |
| 0x00000010 | 0.5196 | 0.6932 |
| 0x80000000 | 0.5166 | 0.6931 |
| 0x00000080 | 0.5086 | 0.6932 |
| 0x00000002 | 0.5084 | 0.6932 |
| 0x00040000 | 0.5072 | 0.6932 |
| 0x00004000 | 0.5060 | 0.6932 |
| 0x00000040 | 0.5046 | 0.6932 |
| 0x00000400 | 0.5046 | 0.6932 |
| 0x08000000 | 0.5038 | 0.6932 |

## Stretch: Δ search at depth 120 (v1 all < 0.515)

Wall clock: 346.7s — top 10:

| Δ | val_acc | loss |
|---|---:|---:|
| 0x00100200 | 0.5104 | 0.6932 |
| 0x00800000 | 0.5102 | 0.6932 |
| 0x84000000 | 0.5102 | 0.6932 |
| 0x00040000 | 0.5100 | 0.6933 |
| 0x00100000 | 0.5098 | 0.6932 |
| 0x00001000 | 0.5076 | 0.6932 |
| 0x00100002 | 0.5076 | 0.6932 |
| 0x00000200 | 0.5066 | 0.6932 |
| 0x00400000 | 0.5066 | 0.6932 |
| 0x04000200 | 0.5062 | 0.6932 |

## Verdict

- Depth 88 best Δ=0x00000020 reached val-acc 0.5198 — **below the 0.55 threshold**. Spatial conv architecture *also* fails to surface signal at depth 88. This tightens the negative result from 'v1 architecture fails' to 'both 1×1 and spatial 3-tap architectures fail' — suggesting the signal horizon is a genuine property of KeeLoq's diffusion at these depths, not an artifact of any one architecture.

