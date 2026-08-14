"""
Week 1 exercise tests for src/rtd/models/rnn.py. These are SUPPOSED to fail
with NotImplementedError until you implement RNNAdapter -- that's not a bug
in the test file, it's the point. Implement rnn.py until these all pass.

Skipped entirely (not failed) if torch isn't installed in your environment.
"""
import pytest

torch = pytest.importorskip("torch")

from rtd.models.rnn import RNNAdapter
from rtd.models.base import HiddenStateOutput


VOCAB_SIZE = 50
BATCH, SEQ = 4, 12


@pytest.fixture
def adapter():
    return RNNAdapter(vocab_size=VOCAB_SIZE, hidden_dim=16, n_layers=2)


@pytest.fixture
def toy_input_ids():
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ))


def test_forward_returns_hidden_state_output(adapter, toy_input_ids):
    out = adapter.forward(toy_input_ids)
    assert isinstance(out, HiddenStateOutput)


def test_logits_shape_matches_vocab(adapter, toy_input_ids):
    out = adapter.forward(toy_input_ids)
    assert out.logits.shape == (BATCH, SEQ, VOCAB_SIZE)


def test_hidden_states_one_per_layer(adapter, toy_input_ids):
    out = adapter.forward(toy_input_ids)
    assert len(out.hidden_states) == 2  # n_layers=2 in the fixture
    for h in out.hidden_states:
        assert h.shape[:2] == (BATCH, SEQ)


def test_attention_or_state_is_hidden_trajectory_not_none(adapter, toy_input_ids):
    state = adapter.get_attention_or_state(toy_input_ids)
    assert state is not None


def test_num_params_non_embedding_smaller_than_total(adapter):
    total = adapter.num_params(non_embedding=False)
    non_embed = adapter.num_params(non_embedding=True)
    assert 0 < non_embed < total


def test_checkpoint_roundtrip_preserves_logits(adapter, toy_input_ids, tmp_path):
    with torch.no_grad():
        logits_before = adapter.forward(toy_input_ids).logits.clone()

    ckpt_dir = tmp_path / "rnn_ckpt"
    adapter.save_checkpoint(ckpt_dir, step=123)

    adapter2 = RNNAdapter(vocab_size=VOCAB_SIZE, hidden_dim=16, n_layers=2)
    meta = adapter2.load_checkpoint(ckpt_dir)

    with torch.no_grad():
        logits_after = adapter2.forward(toy_input_ids).logits

    assert meta.get("step") == 123
    assert torch.allclose(logits_before, logits_after, atol=1e-5)
