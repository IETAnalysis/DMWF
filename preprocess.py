from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def packet_features(
    path: Path, sequence_length: int, length_normalizer: float, clip_value: float
) -> np.ndarray:
    direction, timestamp, length = [], [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required = {"length", "relative_time", "direction"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} does not use the collector CSV schema")
        for row in reader:
            length.append(abs(float(row["length"])) / length_normalizer)
            timestamp.append(float(row["relative_time"]))
            direction.append(1.0 if float(row["direction"]) > 0.5 else -1.0)
    if not direction:
        raise ValueError(f"{path} contains no packet rows")
    timestamp_array = np.asarray(timestamp, dtype=np.float32)
    iat = np.diff(timestamp_array, prepend=timestamp_array[:1])
    iat = np.maximum(iat, 0.0)
    features = np.stack(
        [
            np.asarray(direction, dtype=np.float32),
            iat,
            np.asarray(length, dtype=np.float32),
        ]
    )
    packet_count = features.shape[1]
    if packet_count > sequence_length:
        indices = np.linspace(0, packet_count - 1, sequence_length, dtype=np.int64)
        features = features[:, indices]
    elif packet_count < sequence_length:
        features = np.pad(features, ((0, 0), (0, sequence_length - packet_count)))
    return np.clip(np.nan_to_num(features), -clip_value, clip_value).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--length-normalizer", type=float, default=0.0)
    parser.add_argument("--clip-value", type=float, default=0.0)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a non-empty JSON list")

    labels = sorted(
        {str(label) for entry in entries for label in entry.get("labels", [])}
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    split_rows = {"train": [], "valid": [], "test": []}
    for entry in entries:
        split = entry["split"]
        if split not in split_rows:
            raise ValueError("split must be train, valid, or test")
        source = (args.manifest.parent / entry["path"]).resolve()
        vector = np.zeros(len(labels), dtype=np.float32)
        for label in entry.get("labels", []):
            vector[label_to_index[str(label)]] = 1.0
        split_rows[split].append(
            (
                packet_features(
                    source,
                    args.sequence_length,
                    args.length_normalizer,
                    args.clip_value,
                ),
                vector,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        if not rows:
            raise ValueError(f"manifest has no {split} entries")
        x = np.stack([row[0] for row in rows])
        y = np.stack([row[1] for row in rows])
        c = y.sum(axis=1).astype(np.int64)
        np.savez_compressed(args.output_dir / f"{split}.npz", X=x, y=y, c=c)
    (args.output_dir / "label_mapping.json").write_text(
        json.dumps(label_to_index, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
