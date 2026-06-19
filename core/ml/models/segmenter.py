import torch
import torch.nn as nn
from .components.fusion import FeatureFusion
from .components.attention import TransformerEncoderLayerWithRoPE


class JazzSegmentationModel(nn.Module):
    def __init__(
            self,
            d_model: int = 128,
            n_heads: int = 4,
            d_ff: int = 512,
            num_layers: int = 4,
            num_classes: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()

        self.fusion = FeatureFusion(
            d_model=d_model,
            num_pitches=130,
            continuous_dim=7
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, batch_dict: dict) -> torch.Tensor:
        discrete = batch_dict['discrete']
        continuous = batch_dict['continuous']
        mask = batch_dict.get('padding_mask', None)

        x = self.fusion(discrete, continuous)

        for layer in self.layers:
            x = layer(x, padding_mask=mask)

        logits = self.head(x)
        return logits
