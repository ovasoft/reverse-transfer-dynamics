import pytest

pytest.importorskip("miditok")
pytest.importorskip("symusic")

from symusic import Score, Track, Note

from rtd.tokenize.remi_tokenizer import RemiTokenizerWrapper


def _write_toy_midi(path):
    s = Score()
    s.tpq = 480
    track = Track(program=0, is_drum=False)
    notes = []
    t = 0
    for pitch in [60, 62, 64, 65, 67, 65, 64, 62, 60]:
        notes.append(Note(time=t, duration=240, pitch=pitch, velocity=80))
        t += 240
    track.notes = notes
    s.tracks.append(track)
    s.dump_midi(str(path))


def test_encode_file_produces_nonempty_token_stream(tmp_path):
    midi_path = tmp_path / "toy.mid"
    _write_toy_midi(midi_path)

    tok = RemiTokenizerWrapper()
    ids = tok.encode_file(midi_path)

    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    assert tok.vocab_size > 0


def test_save_and_reload_params(tmp_path):
    tok = RemiTokenizerWrapper()
    params_path = tmp_path / "remi_params.json"
    tok.save_params(params_path)
    assert params_path.exists()

    tok2 = RemiTokenizerWrapper.from_saved(params_path)
    assert tok2.vocab_size == tok.vocab_size
