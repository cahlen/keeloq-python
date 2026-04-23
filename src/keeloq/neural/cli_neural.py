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
