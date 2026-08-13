from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DoQDataset
from .metrics import (
    evaluate_predictions,
    predict_cardinality,
    predict_threshold,
)
from .model import DMWFModel


def evaluate(args) -> dict:
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not {"model_state", "model_config", "decision_threshold"}.issubset(checkpoint):
        raise ValueError("checkpoint is not a DMWF release checkpoint")
    model = DMWFModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = DoQDataset(args.test_file)
    if dataset.num_classes != model.num_classes:
        raise ValueError("test labels do not match checkpoint classes")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    labels, scores, predicted_counts = [], [], []
    with torch.no_grad():
        for signals, targets, _ in loader:
            logits, cardinality_logits, *_ = model(signals.to(device))
            labels.append(targets.numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            predicted_counts.append(torch.argmax(cardinality_logits, dim=-1).cpu().numpy())
    y_true = np.concatenate(labels).astype(np.float32)
    y_score = np.concatenate(scores).astype(np.float32)
    predicted_cardinality = np.concatenate(predicted_counts).astype(np.int64)
    threshold = float(checkpoint["decision_threshold"])

    threshold_prediction = predict_threshold(y_score, threshold)
    native_prediction = predict_cardinality(
        y_score, predicted_cardinality, max_cardinality=model.max_cardinality
    )
    result = {
        "model": "DMWF",
        "scenario": args.scenario,
        "ranking_k": args.ranking_k,
        "frozen_validation_threshold": threshold,
        "global_threshold": evaluate_predictions(
            y_true,
            y_score,
            threshold_prediction,
            ranking_k=args.ranking_k,
            scenario=args.scenario,
        ),
        "native_cardinality_diagnostic": evaluate_predictions(
            y_true,
            y_score,
            native_prediction,
            ranking_k=args.ranking_k,
            scenario=args.scenario,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--scenario", choices=["closed", "open"], required=True)
    parser.add_argument("--ranking-k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="auto")
    result = evaluate(parser.parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
