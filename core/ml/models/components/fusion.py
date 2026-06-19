import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(
            self,
            d_model: int = 128,
            num_pitches: int = 130,
            continuous_dim: int = 7
    ):
        super().__init__()

        self.pitch_embed = nn.Embedding(
            num_embeddings=num_pitches,
            embedding_dim=64,
            padding_idx=0
        )

        self.continuous_proj = nn.Sequential(
            nn.Linear(continuous_dim, 32),
            nn.GELU()
        )

        concat_dim = 64 + 32

        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )

    def forward(self, discrete_tensor: torch.Tensor, continuous_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            discrete_tensor: (batch_size, seq_len, 1)
            continuous_tensor: (batch_size, seq_len, 7)
        Returns:
            Tensor (batch_size, seq_len, d_model)
        """

        pitch_indices = discrete_tensor.squeeze(-1)

        pitch_emb = self.pitch_embed(pitch_indices)

        cont_emb = self.continuous_proj(continuous_tensor)

        fused = torch.cat([pitch_emb, cont_emb], dim=-1)

        return self.output_proj(fused)
    