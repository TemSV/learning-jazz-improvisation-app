import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW


from core.ml.models.segmenter import JazzSegmentationModel
from core.ml.losses.masked_loss import MaskedFocalLoss
from core.ml.data.dataset import JazzSequenceDataset
from core.ml.data.collate import transformer_collate_fn


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load("data/cache/dataset_cache.npz")

    train_dataset = JazzSequenceDataset(
        features=data['X_train'],
        labels=data['y_train'],
        discrete_cols_idx=[1],
        continuous_cols_idx=list(range(2, 9))
    )
    val_dataset = JazzSequenceDataset(
        features=data['X_val'],
        labels=data['y_val'],
        discrete_cols_idx=[1],
        continuous_cols_idx=list(range(2, 9))
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=transformer_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=transformer_collate_fn
    )

    model = JazzSegmentationModel().to(device)
    criterion = MaskedFocalLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            if batch is None:
                continue

            discrete = batch['discrete'].to(device)
            continuous = batch['continuous'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['padding_mask'].to(device)

            optimizer.zero_grad()
            logits = model({"discrete": discrete, "continuous": continuous, "padding_mask": mask})
            loss = criterion(logits, labels, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                discrete = batch['discrete'].to(device)
                continuous = batch['continuous'].to(device)
                labels = batch['labels'].to(device)
                mask = batch['padding_mask'].to(device)

                logits = model({"discrete": discrete, "continuous": continuous, "padding_mask": mask})
                loss = criterion(logits, labels, mask)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(train_loader):.4f} "
            f"| Val Loss: {val_loss / len(val_loader):.4f}")

        checkpoint_dir = "C:\polytech\Diploma\learning-jazz-improvisation-app\checkpoints"
        save_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()