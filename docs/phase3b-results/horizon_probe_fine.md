# Phase 3b Horizon Probe — Fine-Grained (57, 58, 59)

Top-8 Δs from depth-56 search, 100 k samples × 2 epochs. Viability threshold: val_acc ≥ 0.55.

## Results

| trained_depth | Δ=0x00000002 | Δ=0x00010000 | Δ=0x00000004 | Δ=0x02000000 | Δ=0x00020000 | Δ=0x04000002 | Δ=0x01000000 | Δ=0x00800000 | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | 0.5162 | 0.5204 | 0.4994 | 0.5078 | 0.5080 | 0.5010 | 0.5386 | 0.5214 | **0.5386** ❌ |
| 58 | 0.5096 | 0.4956 | 0.5034 | 0.5180 | 0.5166 | 0.5022 | 0.5152 | 0.5340 | **0.5340** ❌ |
| 59 | 0.5088 | 0.5048 | 0.5100 | 0.5344 | 0.4976 | 0.4980 | 0.5122 | 0.5298 | **0.5344** ❌ |

## Conclusion

No fine-grained depth crossed the viability threshold. The cliff is very sharp — at or just past depth 56. d64.pt's trained depth 56 is the maximum viable point with this architecture + budget.

