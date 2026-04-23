# Phase 3b Horizon Probe

Tests the signal cliff between depth 56 (known signal, d64.pt trained here) and depth 88 (known collapse). Budget per cell: 100 k samples × 2 epochs, depth-2/width-16 tiny model. Viability threshold: val_acc ≥ 0.55.

## Results (val-accuracy at tiny-budget training)

| trained_depth | Δ=0x00000002 | Δ=0x00010000 | Δ=0x00800000 | best |
|---:|---:|---:|---:|---:|
| 60 | 0.5054 | 0.5016 | 0.5282 | **0.5282** ❌ |
| 64 | 0.5110 | 0.5022 | 0.5070 | **0.5110** ❌ |
| 68 | 0.4912 | 0.4950 | 0.5088 | **0.5088** ❌ |
| 72 | 0.5016 | 0.5008 | 0.4992 | **0.5016** ❌ |
| 76 | 0.5150 | 0.5032 | 0.5146 | **0.5150** ❌ |
| 80 | 0.4930 | 0.5052 | 0.4920 | **0.5052** ❌ |

## Conclusion

No probe depth crossed the viability threshold. The cliff is below depth 60; the d64.pt floor (trained at 56) is effectively the maximum viable trained depth with this architecture and data budget.

