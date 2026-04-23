"""CLI tests for `keeloq neural ...` subcommands."""

from __future__ import annotations

import contextlib
import json

import pytest

with contextlib.suppress(ImportError):
    import torch  # noqa: F401

from typer.testing import CliRunner

from keeloq.cli import app

runner = CliRunner()


@pytest.mark.gpu
def test_neural_train_command(tmp_path) -> None:
    ckpt = tmp_path / "tiny.pt"
    result = runner.invoke(
        app,
        [
            "neural",
            "train",
            "--rounds",
            "1",
            "--delta",
            "0x80000000",
            "--samples",
            "4000",
            "--batch-size",
            "512",
            "--epochs",
            "1",
            "--depth",
            "2",
            "--width",
            "8",
            "--seed",
            "0",
            "--out",
            str(ckpt),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert ckpt.exists()


@pytest.mark.gpu
def test_neural_evaluate_command(tmp_path) -> None:
    ckpt = tmp_path / "tiny.pt"
    train_res = runner.invoke(
        app,
        [
            "neural",
            "train",
            "--rounds",
            "1",
            "--delta",
            "0x80000000",
            "--samples",
            "4000",
            "--batch-size",
            "512",
            "--epochs",
            "1",
            "--depth",
            "2",
            "--width",
            "8",
            "--seed",
            "0",
            "--out",
            str(ckpt),
        ],
    )
    assert train_res.exit_code == 0

    eval_res = runner.invoke(
        app,
        [
            "neural",
            "evaluate",
            "--checkpoint",
            str(ckpt),
            "--rounds",
            "1",
            "--samples",
            "2000",
            "--seed",
            "42",
        ],
    )
    assert eval_res.exit_code == 0, eval_res.stdout
    data = json.loads(eval_res.stdout)
    assert "accuracy" in data and 0.0 <= data["accuracy"] <= 1.0
