"""
Tests for the reference TransformerAdapter (src/rtd/models/transformer.py).
This is the "golden path" adapter -- get these tests green first, in an
environment with torch installed, before trusting anything built on top of
it (the trainer, the eval harness wrapper, etc).

Skipped entirely if torch/transformers aren't installed.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from rtd.models.transformer import TransformerAdapter, build_gpt2_reference, TOY_CONFIG


VOCAB_SIZE = 100
BATCH, SEQ = 2, 16


@pytest.fixture
def adapter():
    return TransformerAdapter(vocab_size=VOCAB_SIZE, config_overrides=TOY_CONFIG)


@pytest.fixture
def toy_input_ids():
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ))


def test_forward_shapes(adapter, toy_input_ids):
    out = adapter.forward(toy_input_ids)
    assert out.logits.shape == (BATCH, SEQ, VOCAB_SIZE)
    # GPT2 returns n_layer + 1 hidden_states (embedding output + each block)
    assert len(out.hidden_states) == TOY_CONFIG["n_layer"] + 1
    assert len(out.attention_or_state) == TOY_CONFIG["n_layer"]
    assert out.attention_or_state[0].shape == (BATCH, TOY_CONFIG["n_head"], SEQ, SEQ)


def test_num_params_non_embedding_smaller(adapter):
    total = adapter.num_params(non_embedding=False)
    non_embed = adapter.num_params(non_embedding=True)
    assert 0 < non_embed < total


def test_checkpoint_roundtrip(adapter, toy_input_ids, tmp_path):
    with torch.no_grad():
        logits_before = adapter.forward(toy_input_ids).logits.clone()

    ckpt_dir = tmp_path / "tf_ckpt"
    adapter.save_checkpoint(ckpt_dir, step=42)

    adapter2 = TransformerAdapter(vocab_size=VOCAB_SIZE, config_overrides=TOY_CONFIG)
    meta = adapter2.load_checkpoint(ckpt_dir)

    with torch.no_grad():
        logits_after = adapter2.forward(toy_input_ids).logits

    assert meta["step"] == 42
    assert torch.allclose(logits_before, logits_after, atol=1e-5)


def test_gpt2_reference_uses_published_small_config():
    gpt2 = build_gpt2_reference(vocab_size=VOCAB_SIZE)
    assert gpt2.model.config.n_layer == 12
    assert gpt2.model.config.n_embd == 768
    assert gpt2.model.config.n_head == 12
