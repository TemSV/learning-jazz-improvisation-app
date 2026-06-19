import torch
import numpy as np
from torch.utils.data import Dataset


class JazzSequenceDataset(Dataset):
    def __init__(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            discrete_cols_idx: list[int],
            continuous_cols_idx: list[int],
            sequence_length: int = 512,
            stride: int = 256
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = []

        if features is None or len(features) == 0:
            raise ValueError("Features array is empty.")
        if len(labels) != len(features):
            raise ValueError(f"Shape mismatch: features ({len(features)}) != labels ({len(labels)}).")

        melids = features[:, 0]
        unique_melids = np.unique(melids)

        for melid in unique_melids:
            mask = (melids == melid)
            mel_features = features[mask]
            mel_labels = labels[mask]

            n_notes = len(mel_features)
            if n_notes < 10:
                continue

            discrete_seq = torch.tensor(mel_features[:, discrete_cols_idx], dtype=torch.long)
            continuous_seq = torch.tensor(mel_features[:, continuous_cols_idx], dtype=torch.float32)
            label_seq = torch.tensor(mel_labels, dtype=torch.long)

            for start_idx in range(0, n_notes, self.stride):
                end_idx = min(start_idx + self.sequence_length, n_notes)

                if end_idx - start_idx < 10:
                    continue

                self.sequences.append({
                    "discrete": discrete_seq[start_idx:end_idx],
                    "continuous": continuous_seq[start_idx:end_idx],
                    "labels": label_seq[start_idx:end_idx]
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of bounds.")
        return self.sequences[idx]
