"""keeloq neural {train,evaluate,recover-key,auto} subcommands.

Mounted under the main `keeloq` app via src/keeloq/cli.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

neural_app = typer.Typer(help="Neural differential cryptanalysis (Phase 3b).")


@neural_app.command("train")
def train_cmd(
    rounds: Annotated[int, typer.Option()],
    delta: Annotated[str, typer.Option(help="Plaintext difference, 0x... hex")],
    samples: Annotated[int, typer.Option()],
    out: Annotated[str, typer.Option(help="Checkpoint output path")],
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4096,
    epochs: Annotated[int, typer.Option()] = 20,
    lr: Annotated[float, typer.Option()] = 2e-3,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 1e-5,
    seed: Annotated[int, typer.Option()] = 0,
    depth: Annotated[int, typer.Option()] = 5,
    width: Annotated[int, typer.Option()] = 512,
) -> None:
    """Train a distinguisher and save a checkpoint."""
    from keeloq.neural.distinguisher import TrainingConfig, save_checkpoint, train

    delta_int = int(delta, 16) if delta.startswith(("0x", "0X")) else int(delta)
    cfg = TrainingConfig(
        rounds=rounds,
        delta=delta_int,
        n_samples=samples,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        depth=depth,
        width=width,
    )
    model, result = train(cfg)
    save_checkpoint(model, result, Path(out))
    typer.echo(
        json.dumps(
            {
                "final_loss": result.final_loss,
                "final_val_accuracy": result.final_val_accuracy,
                "wall_time_s": result.wall_time_s,
                "checkpoint": out,
            }
        )
    )


@neural_app.command("evaluate")
def evaluate_cmd(
    checkpoint: Annotated[str, typer.Option()],
    rounds: Annotated[int, typer.Option()],
    samples: Annotated[int, typer.Option()] = 1_000_000,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Evaluate a distinguisher checkpoint and emit metrics as JSON."""
    from keeloq.neural.distinguisher import load_checkpoint
    from keeloq.neural.evaluation import evaluate

    model, result = load_checkpoint(Path(checkpoint))
    report = evaluate(
        model,
        rounds=rounds,
        delta=result.config.delta,
        n_samples=samples,
        seed=seed,
    )
    typer.echo(json.dumps(asdict(report)))


from keeloq._types import bits_to_int, int_to_bits  # noqa: E402


def _parse_bitvec_32(s: str, name: str) -> int:
    s = s.strip()
    if s.startswith(("0x", "0X")):
        v = int(s, 16)
        if v.bit_length() > 32:
            raise typer.BadParameter(f"{name} doesn't fit in 32 bits")
        return v
    if len(s) != 32:
        raise typer.BadParameter(f"{name} must be 32 bits (got {len(s)})")
    return bits_to_int(s)


def _parse_colon_pair(arg: str, label: str) -> tuple[int, int]:
    if ":" not in arg:
        raise typer.BadParameter(f"--{label} must be 'left:right' (got {arg!r})")
    left_s, right_s = arg.split(":", 1)
    return (
        _parse_bitvec_32(left_s, f"{label} first half"),
        _parse_bitvec_32(right_s, f"{label} second half"),
    )


_STATUS_EXIT = {
    "SUCCESS": 0,
    "CRASH": 1,
    "WRONG_KEY": 2,
    "UNSAT": 3,
    "TIMEOUT": 4,
    "NEURAL_FAIL": 5,
    "BACKTRACK_EXHAUSTED": 6,
}


@neural_app.command("recover-key")
def recover_key_cmd(
    checkpoint: Annotated[str, typer.Option()],
    rounds: Annotated[int, typer.Option()],
    diff_pair: Annotated[
        list[str],
        typer.Option(
            "--diff-pair",
            help="Differential pair 'c0:c1' (both ciphertexts), repeatable. "
            "Used by the neural distinguisher.",
        ),
    ],
    sat_pair: Annotated[
        list[str],
        typer.Option(
            "--sat-pair",
            help="SAT constraint pair 'plaintext:ciphertext', repeatable. "
            "Used to finish the attack after neural prefix.",
        ),
    ],
    beam_width: Annotated[int, typer.Option("--beam-width")] = 8,
    neural_target_bits: Annotated[int | None, typer.Option("--neural-target-bits")] = None,
    sat_timeout: Annotated[float, typer.Option("--sat-timeout")] = 60.0,
    max_backtracks: Annotated[int, typer.Option("--max-backtracks")] = 8,
    key_hint: Annotated[
        list[str] | None,
        typer.Option("--key-hint", help="Extra K-bit hint 'index:value', repeatable."),
    ] = None,
) -> None:
    """Run hybrid_attack with a pre-trained distinguisher."""
    from keeloq.neural.distinguisher import load_checkpoint
    from keeloq.neural.hybrid import hybrid_attack

    if not diff_pair:
        raise typer.BadParameter("at least one --diff-pair is required")
    if not sat_pair:
        raise typer.BadParameter("at least one --sat-pair is required")

    diff_pairs = [_parse_colon_pair(p, "diff-pair") for p in diff_pair]
    sat_pairs = [_parse_colon_pair(p, "sat-pair") for p in sat_pair]

    extra_hints: dict[int, int] = {}
    for kh in key_hint or []:
        if ":" not in kh:
            raise typer.BadParameter(f"--key-hint must be 'index:value' (got {kh!r})")
        idx_s, val_s = kh.split(":", 1)
        idx, val = int(idx_s), int(val_s)
        if not 0 <= idx < 64 or val not in (0, 1):
            raise typer.BadParameter(f"bad --key-hint {kh!r}")
        extra_hints[idx] = val

    model, _ = load_checkpoint(Path(checkpoint))
    result = hybrid_attack(
        rounds=rounds,
        pairs=diff_pairs,
        sat_pairs=sat_pairs,
        distinguisher=model,
        beam_width=beam_width,
        neural_target_bits=neural_target_bits,
        sat_timeout_s=sat_timeout,
        max_backtracks=max_backtracks,
        extra_key_hints=extra_hints or None,
    )

    typer.echo(
        json.dumps(
            {
                "status": result.status,
                "recovered_key": (
                    int_to_bits(result.recovered_key, 64)
                    if result.recovered_key is not None
                    else None
                ),
                "bits_recovered_neurally": result.bits_recovered_neurally,
                "neural_wall_time_s": result.neural_wall_time_s,
                "sat_wall_time_s": result.sat_wall_time_s,
                "verify_result": result.verify_result,
            }
        )
    )
    raise typer.Exit(code=_STATUS_EXIT[result.status])


@neural_app.command("auto")
def auto_cmd(
    rounds: Annotated[int, typer.Option(help="Attack depth")],
    samples: Annotated[int, typer.Option()] = 10_000_000,
    trained_depth: Annotated[
        int | None,
        typer.Option(
            "--trained-depth",
            help="Depth at which to train the distinguisher. Default: rounds // 2.",
        ),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4096,
    epochs: Annotated[int, typer.Option()] = 20,
    lr: Annotated[float, typer.Option()] = 2e-3,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 1e-5,
    seed: Annotated[int, typer.Option()] = 0,
    depth: Annotated[int, typer.Option()] = 5,
    width: Annotated[int, typer.Option()] = 512,
    pairs: Annotated[int, typer.Option()] = 512,
    checkpoint_out: Annotated[str, typer.Option("--checkpoint-out")] = "checkpoints/d_auto.pt",
    delta_search_budget: Annotated[int, typer.Option("--delta-search-budget")] = 200_000,
) -> None:
    """End-to-end: Δ search → train → evaluate → attack on synthetic pairs."""
    import time

    import torch

    from keeloq.gpu_cipher import encrypt_batch
    from keeloq.neural.differences import search_delta
    from keeloq.neural.distinguisher import (
        TrainingConfig,
        save_checkpoint,
        train,
    )
    from keeloq.neural.evaluation import evaluate
    from keeloq.neural.hybrid import hybrid_attack

    if trained_depth is None:
        trained_depth = max(4, rounds // 2)

    neural_bits = rounds - trained_depth

    t_all = time.perf_counter()

    # Δ search AT the trained depth (the depth where we'll actually run the
    # distinguisher during recovery).
    cands = search_delta(
        rounds=trained_depth,
        candidates=None,
        tiny_budget_samples=delta_search_budget,
        tiny_budget_epochs=1,
        seed=seed,
    )
    best_delta = cands[0].delta

    cfg = TrainingConfig(
        rounds=trained_depth,
        delta=best_delta,
        n_samples=samples,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        depth=depth,
        width=width,
    )
    model, train_result = train(cfg)
    save_checkpoint(model, train_result, Path(checkpoint_out))

    report = evaluate(
        model,
        rounds=trained_depth,
        delta=best_delta,
        n_samples=min(100_000, samples),
        seed=seed + 1,
    )

    target_key = 0xA1B2_C3D4_E5F6_0718
    gen = torch.Generator(device="cpu").manual_seed(seed + 2)
    pts0 = torch.randint(0, 1 << 32, (pairs,), generator=gen, dtype=torch.int64).to(
        dtype=torch.uint32, device="cuda"
    )
    # CUDA uint32 XOR unsupported — apply delta on CPU first.
    pts0_cpu = pts0.cpu()
    delta_t = torch.tensor(best_delta, dtype=torch.uint32)
    pts1 = (pts0_cpu ^ delta_t).to("cuda")
    keys = torch.tensor(
        [[target_key & 0xFFFFFFFF, (target_key >> 32) & 0xFFFFFFFF]] * pairs,
        dtype=torch.uint32,
        device="cuda",
    )
    c0 = encrypt_batch(pts0, keys, rounds=rounds)
    c1 = encrypt_batch(pts1, keys, rounds=rounds)
    diff_pairs = [
        (int(c0[i].item()) & 0xFFFFFFFF, int(c1[i].item()) & 0xFFFFFFFF) for i in range(pairs)
    ]
    # SAT pairs: each is (plaintext_a, ciphertext_a) for the p0 side.
    sat_pairs_list = [
        (int(pts0_cpu[i].item()) & 0xFFFFFFFF, int(c0[i].item()) & 0xFFFFFFFF) for i in range(pairs)
    ]

    # If the cyclic schedule leaves bits beyond `rounds` unconstrained,
    # synthesize extra_hints covering them.
    extra_hints: dict[int, int] | None = None
    if rounds < 64:
        extra_hints = {i: (target_key >> (63 - i)) & 1 for i in range(rounds, 64)}

    att = hybrid_attack(
        rounds=rounds,
        pairs=diff_pairs[: min(8, len(diff_pairs))],
        sat_pairs=sat_pairs_list[: min(8, len(sat_pairs_list))],
        distinguisher=model,
        beam_width=8,
        neural_target_bits=neural_bits,
        sat_timeout_s=120.0,
        extra_key_hints=extra_hints,
    )

    typer.echo(
        json.dumps(
            {
                "wall_time_total_s": time.perf_counter() - t_all,
                "trained_depth": trained_depth,
                "attack_depth": rounds,
                "delta": f"0x{best_delta:08x}",
                "train": {
                    "final_val_accuracy": train_result.final_val_accuracy,
                    "final_loss": train_result.final_loss,
                    "wall_time_s": train_result.wall_time_s,
                },
                "evaluate": asdict(report),
                "attack_status": att.status,
                "attack_recovered_key": (
                    int_to_bits(att.recovered_key, 64) if att.recovered_key is not None else None
                ),
                "attack_expected_key": int_to_bits(target_key, 64),
                "bits_recovered_neurally": att.bits_recovered_neurally,
                "neural_wall_time_s": att.neural_wall_time_s,
                "sat_wall_time_s": att.sat_wall_time_s,
                "checkpoint": checkpoint_out,
            }
        )
    )
    raise typer.Exit(code=_STATUS_EXIT[att.status])
