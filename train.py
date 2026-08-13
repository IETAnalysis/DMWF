from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DoQDataset
from .metrics import select_global_threshold
from .model import DMWFModel, slot_diversity_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def collect_predictions(model, loader, device):
    model.eval()
    labels, probabilities, cardinalities = [], [], []
    with torch.no_grad():
        for signals, targets, _ in loader:
            logits, card_logits, *_ = model(signals.to(device))
            labels.append(targets.numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            cardinalities.append(torch.argmax(card_logits, dim=-1).cpu().numpy())
    return (
        np.concatenate(labels).astype(np.float32),
        np.concatenate(probabilities).astype(np.float32),
        np.concatenate(cardinalities).astype(np.int64),
    )


def train(args) -> Path:
    set_seed(args.seed)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = DoQDataset(data_dir / "train.npz")
    valid_data = DoQDataset(data_dir / "valid.npz")
    if (
        train_data.num_classes != valid_data.num_classes
        or train_data.sequence_length != valid_data.sequence_length
    ):
        raise ValueError("training and validation schemas do not match")
    if int(train_data.c.max()) > args.max_cardinality:
        raise ValueError("max_cardinality is smaller than a training label count")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=len(train_data) >= args.batch_size,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_config = {
        "num_classes": train_data.num_classes,
        "max_cardinality": args.max_cardinality,
        "sequence_length": train_data.sequence_length,
        "max_slots": args.max_slots,
        "embedding_dimension": args.embedding_dimension,
        "attention_heads": args.attention_heads,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
        "auxiliary_weight": args.auxiliary_fusion_weight,
        "statistics_weight": args.statistics_fusion_weight,
        "lctr_clip_value": args.lctr_clip_value,
        "lctr_epsilon": args.lctr_epsilon,
        "first_branch_channels": args.first_branch_channels,
        "second_branch_channels": args.second_branch_channels,
        "convolution_kernel_size": args.convolution_kernel_size,
        "se_reduction": args.se_reduction,
        "pool_kernel": args.pool_kernel,
        "pool_stride": args.pool_stride,
        "feedforward_multiplier": args.feedforward_multiplier,
        "statistics_pool_bins": args.statistics_pool_bins,
        "statistics_hidden_dimension": args.statistics_hidden_dimension,
        "positional_encoding_base": args.positional_encoding_base,
        "positional_max_length": args.positional_max_length,
        "slot_query_scale": args.slot_query_scale,
        "noisy_or_epsilon": args.noisy_or_epsilon,
    }
    model = DMWFModel(**model_config).to(device)
    positive_weight = torch.full(
        [train_data.num_classes], args.positive_weight, device=device
    )
    classification = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    cardinality = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    patience = args.patience
    history = []
    checkpoint_path = output_dir / "best.pt"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for signals, targets, counts in tqdm(train_loader, leave=False):
            signals = signals.to(device)
            targets = targets.to(device)
            counts = counts.to(device)
            optimizer.zero_grad(set_to_none=True)
            (
                logits,
                cardinality_logits,
                auxiliary_logits,
                statistics_logits,
                _slot_logits,
                slots,
            ) = model(signals)
            loss = (
                classification(logits, targets)
                + args.auxiliary_loss_weight * classification(auxiliary_logits, targets)
                + args.statistics_loss_weight * classification(statistics_logits, targets)
                + args.cardinality_loss_weight * cardinality(cardinality_logits, counts)
                + args.slot_diversity_weight * slot_diversity_loss(slots)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            running_loss += float(loss.item()) * len(signals)
            sample_count += len(signals)

        y_true, y_score, _ = collect_predictions(model, valid_loader, device)
        selected = select_global_threshold(y_true, y_score)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(sample_count, 1),
            "validation_micro_f1": selected.micro_f1,
            "validation_threshold": selected.threshold,
        }
        history.append(row)
        print(json.dumps(row))
        if selected.micro_f1 > best_f1:
            best_f1 = selected.micro_f1
            best_epoch = epoch
            best_threshold = selected.threshold
            patience = args.patience
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "decision_threshold": best_threshold,
                    "scenario": args.scenario,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            patience -= 1
        scheduler.step()
        if patience <= 0:
            break

    summary = {
        "model": "DMWF",
        "scenario": args.scenario,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_micro_f1": best_f1,
        "decision_threshold": best_threshold,
        "model_config": model_config,
        "training": {
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "learning_rate": args.learning_rate,
            "positive_weight": args.positive_weight,
            "cardinality_loss_weight": args.cardinality_loss_weight,
            "slot_diversity_weight": args.slot_diversity_weight,
        },
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return checkpoint_path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("--data-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--scenario", choices=["closed", "open"], required=True)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--epochs", type=int, default=0)
    command.add_argument("--patience", type=int, default=0)
    command.add_argument("--batch-size", type=int, default=0)
    command.add_argument("--eval-batch-size", type=int, default=0)
    command.add_argument("--learning-rate", type=float, default=0.0)
    command.add_argument("--weight-decay", type=float, default=0.0)
    command.add_argument("--positive-weight", type=float, default=0.0)
    command.add_argument("--cardinality-loss-weight", type=float, default=0.0)
    command.add_argument("--slot-diversity-weight", type=float, default=0.0)
    command.add_argument("--auxiliary-loss-weight", type=float, default=0.0)
    command.add_argument("--statistics-loss-weight", type=float, default=0.0)
    command.add_argument("--auxiliary-fusion-weight", type=float, default=0.0)
    command.add_argument("--statistics-fusion-weight", type=float, default=0.0)
    command.add_argument("--max-cardinality", type=int, default=0)
    command.add_argument("--max-slots", type=int, default=0)
    command.add_argument("--embedding-dimension", type=int, default=0)
    command.add_argument("--attention-heads", type=int, default=0)
    command.add_argument("--encoder-layers", type=int, default=0)
    command.add_argument("--decoder-layers", type=int, default=0)
    command.add_argument("--dropout", type=float, default=0.0)
    command.add_argument("--gradient-clip", type=float, default=0.0)
    command.add_argument("--lctr-clip-value", type=float, default=0.0)
    command.add_argument("--lctr-epsilon", type=float, default=0.0)
    command.add_argument("--first-branch-channels", type=int, default=0)
    command.add_argument("--second-branch-channels", type=int, default=0)
    command.add_argument("--convolution-kernel-size", type=int, default=0)
    command.add_argument("--se-reduction", type=int, default=0)
    command.add_argument("--pool-kernel", type=int, default=0)
    command.add_argument("--pool-stride", type=int, default=0)
    command.add_argument("--feedforward-multiplier", type=int, default=0)
    command.add_argument("--statistics-pool-bins", type=int, default=0)
    command.add_argument("--statistics-hidden-dimension", type=int, default=0)
    command.add_argument("--positional-encoding-base", type=float, default=0.0)
    command.add_argument("--positional-max-length", type=int, default=0)
    command.add_argument("--slot-query-scale", type=float, default=0.0)
    command.add_argument("--noisy-or-epsilon", type=float, default=0.0)
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--device", default="auto")
    return command


def main() -> int:
    checkpoint = train(parser().parse_args())
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
