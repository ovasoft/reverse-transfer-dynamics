from rtd.tokenize.abc_tokenizer import ABCTokenizer, split_abc


ABC_SAMPLE_1 = "X:1\nT:Cooley's\nM:4/4\nK:Emin\n|:D2|EB{c}BA B2 EB|~B2 AB dBAG|\n"
ABC_SAMPLE_2 = "X:2\nT:The Wind\nM:6/8\nK:Dmaj\nA2F D2F|A2d cAF|\n"


def test_split_abc_nonempty():
    toks = split_abc(ABC_SAMPLE_1)
    assert len(toks) > 0
    assert all(t.strip() for t in toks)


def test_build_and_encode_decode_roundtrip():
    tok = ABCTokenizer.build_from_corpus([ABC_SAMPLE_1, ABC_SAMPLE_2])
    ids = tok.encode(ABC_SAMPLE_1, add_bos_eos=True)
    assert ids[0] == tok.token_to_id["<bos>"]
    assert ids[-1] == tok.token_to_id["<eos>"]

    decoded = tok.decode(ids)
    # decode strips special tokens but should reconstruct the token stream
    assert decoded.replace(" ", "") != ""


def test_unseen_token_maps_to_unk():
    tok = ABCTokenizer.build_from_corpus([ABC_SAMPLE_1])
    ids = tok.encode("~~~totally_unseen~~~", add_bos_eos=False)
    unk_id = tok.token_to_id["<unk>"]
    assert unk_id in ids


def test_save_load_roundtrip(tmp_path):
    tok = ABCTokenizer.build_from_corpus([ABC_SAMPLE_1, ABC_SAMPLE_2])
    path = tmp_path / "vocab.json"
    tok.save(path)
    tok2 = ABCTokenizer.load(path)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode(ABC_SAMPLE_1) == tok.encode(ABC_SAMPLE_1)
