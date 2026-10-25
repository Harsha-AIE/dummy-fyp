import torch

from edukg.agents.knowledge_tracing.dkt_model import DKTModel


def test_forward_output_shape_and_range() -> None:
    torch.manual_seed(0)
    vocab_size = 5
    model = DKTModel(vocab_size=vocab_size, d_model=8, n_heads=2, n_layers=1)
    interaction_ids = torch.tensor([[0, 6, 2, 8]])  # concept idx, some flagged "correct" (+vocab_size)

    probs = model(interaction_ids)

    assert probs.shape == (1, 4, vocab_size)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_attention_weights_shape_and_causal_structure() -> None:
    torch.manual_seed(0)
    vocab_size = 4
    model = DKTModel(vocab_size=vocab_size, d_model=8, n_heads=2, n_layers=1)
    interaction_ids = torch.tensor([[0, 1, 2]])

    weights = model.attention_weights(interaction_ids)

    assert weights.shape == (1, 3, 3)
    # Causal mask: position 0 can only attend to itself, so its weight mass
    # outside index 0 should be ~0.
    assert weights[0, 0, 1].item() == 0.0
    assert weights[0, 0, 2].item() == 0.0


def test_forward_is_deterministic_in_eval_mode() -> None:
    torch.manual_seed(1)
    model = DKTModel(vocab_size=3, d_model=8, n_heads=2, n_layers=1)
    model.eval()
    interaction_ids = torch.tensor([[0, 4, 1]])

    with torch.no_grad():
        first = model(interaction_ids)
        second = model(interaction_ids)

    assert torch.allclose(first, second)
