import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from core.ml.models.segmenter import JazzSegmentationModel
from core.ml.data.dataset import JazzSequenceDataset
from core.ml.data.collate import transformer_collate_fn


def calculate_window_metrics(pred_seq: np.ndarray, true_seq: np.ndarray, tolerance: int = 1) -> tuple[int, int, int]:
    pred_idx = np.where(pred_seq == 1)[0]
    true_idx = np.where(true_seq == 1)[0]

    tp, fp = 0, 0
    matched_true = set()

    for p in pred_idx:
        valid_targets = [t for t in true_idx if abs(t - p) <= tolerance and t not in matched_true]

        if valid_targets:
            closest_t = min(valid_targets, key=lambda t: abs(t - p))
            matched_true.add(closest_t)
            tp += 1
        else:
            fp += 1

    fn = len(true_idx) - len(matched_true)
    return tp, fp, fn


def evaluate(cache_path: str, model_path: str, tolerance: int, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Dataset cache not found: {cache_path}")

    data = np.load(cache_path)

    test_dataset = JazzSequenceDataset(
        features=data['X_test'],
        labels=data['y_test'],
        discrete_cols_idx=[1],
        continuous_cols_idx=list(range(2, 9))
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=transformer_collate_fn
    )

    model = JazzSegmentationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            discrete = batch['discrete'].to(device)
            continuous = batch['continuous'].to(device)
            labels = batch['labels'].numpy()
            mask = batch['padding_mask'].numpy()

            logits = model(
                {"discrete": discrete, "continuous": continuous, "padding_mask": batch['padding_mask'].to(device)})

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()

            for b in range(preds.shape[0]):
                seq_len = int(mask[b].sum())
                pred_seq = preds[b, :seq_len]
                true_seq = labels[b, :seq_len]

                tp, fp, fn = calculate_window_metrics(pred_seq, true_seq, tolerance)
                total_tp += tp
                total_fp += fp
                total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tolerance": tolerance
    }

    print(json.dumps(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Jazz Segmentation Model")
    parser.add_argument("--cache-path", type=str, required=True, help="Path to dataset_cache.npz")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--tolerance", type=int, default=1, help="Window tolerance for metric calculation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for DataLoader")

    args = parser.parse_args()

    evaluate(
        cache_path=args.cache_path,
        model_path=args.model_path,
        tolerance=args.tolerance,
        batch_size=args.batch_size
    )
