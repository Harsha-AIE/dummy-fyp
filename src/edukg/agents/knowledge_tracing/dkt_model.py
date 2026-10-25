from __future__ import annotations

import torch
from torch import nn

_MAX_POSITIONS = 512


class DKTModel(nn.Module):
    """Small self-attentive (SAKT-style) Deep Knowledge Tracing model.

    Deliberately small: this project's current interaction volume is
    nowhere near what would justify a large model, and a bigger one would
    just overfit the synthetic bootstrap data faster (see
    scripts/train_dkt.py). Sized to grow with real concept/interaction
    counts later, not fixed at production scale now.

    Input per timestep is a `(concept_idx, correct)` pair, encoded as the
    standard DKT interaction id `concept_idx + vocab_size * correct` (Piech
    et al., 2015), embedded and fed through a single causally-masked
    transformer encoder layer — each timestep can only attend to itself and
    earlier timesteps, matching the "predict the next response from past
    ones" framing knowledge tracing is built around. Output: per-timestep
    P(correct) for every concept in the vocabulary; a concept's mastery is
    read off its output column at the sequence's last timestep.
    """

    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 2, n_layers: int = 1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.interaction_embedding = nn.Embedding(2 * vocab_size, d_model)
        self.position_embedding = nn.Embedding(_MAX_POSITIONS, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.readout = nn.Linear(d_model, vocab_size)

    def _embed(self, interaction_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = interaction_ids.shape
        positions = torch.arange(seq_len, device=interaction_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.interaction_embedding(interaction_ids) + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=interaction_ids.device)
        return x, causal_mask

    def forward(self, interaction_ids: torch.Tensor) -> torch.Tensor:
        """interaction_ids: (batch, seq_len) long tensor of
        `concept_idx + vocab_size*correct` codes. Returns (batch, seq_len,
        vocab_size) sigmoid probabilities — P(correct) per concept at each
        timestep."""
        x, causal_mask = self._embed(interaction_ids)
        encoded = self.encoder(x, mask=causal_mask)
        return torch.sigmoid(self.readout(encoded))

    def attention_weights(self, interaction_ids: torch.Tensor) -> torch.Tensor:
        """Re-runs the first encoder layer's self-attention directly to
        recover per-timestep attention weights, used to identify which
        prior events most influenced a prediction ("top contributing
        events"). Returns (batch, seq_len, seq_len)."""
        x, causal_mask = self._embed(interaction_ids)
        layer = self.encoder.layers[0]
        _, weights = layer.self_attn(x, x, x, attn_mask=causal_mask, need_weights=True, average_attn_weights=True)
        return weights
