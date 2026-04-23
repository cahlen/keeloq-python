# Phase 3b Trained Distinguisher Checkpoints

This directory holds trained neural distinguishers for the Gohr-style attack
pipeline. Each `.pt` file is a PyTorch state-dict + training metadata, produced
by `keeloq neural train`.

## Provenance

Each checkpoint is reproducible from:
- The fixed Δ recorded in `docs/phase3b-results/delta_search.md`
- The `TrainingConfig` (rounds, samples, batch_size, epochs, lr, weight_decay, seed)
  captured inside the checkpoint
- Phase 1's GPU cipher (`keeloq.gpu_cipher.encrypt_batch`) for training-data generation

## Files

- `d64.pt` — distinguisher for 64-round KeeLoq. Floor commitment. Target: `hybrid_attack`
  recovers the full 64-bit key with this checkpoint in <5 min wall clock.
- `d96.pt` — distinguisher for 96-round KeeLoq. Intermediate stepping stone toward
  the 128-round ambition target.
- `d128.pt` — distinguisher for 128-round KeeLoq. Ambition target.

## Reproduction

    # Train a distinguisher (~30 min per depth):
    uv run keeloq neural train --rounds 128 --delta 0x00000001 \
      --samples 10000000 --out checkpoints/d128.pt

    # Verify:
    uv run keeloq neural evaluate --checkpoint checkpoints/d128.pt --rounds 128

Experimental / superseded checkpoints live outside the repo (see `scratch/`).
