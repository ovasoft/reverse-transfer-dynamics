from rtd.eval.niah import (
    build_niah_document,
    build_niah_suite,
    score_retrieval,
    build_accuracy_matrix,
    DEFAULT_LENGTHS,
    DEFAULT_DEPTHS_PCT,
)


class _WhitespaceTokenizer:
    """Minimal word-level tokenizer satisfying niah.py's duck-typed
    interface (.encode/.decode). NIAH operates on plain RefinedWeb text
    (Section 4, Step 1), not ABC notation, so ABCTokenizer -- which is
    tuned for ABC syntax and doesn't preserve whitespace on decode -- is
    the wrong tool here even for tests. The real Phase 2 BPE tokenizer
    (src/rtd/tokenize/bpe_expand.py) is what production NIAH runs should
    use; this stand-in just needs to round-trip words with spaces intact
    so the tests below can check substring containment sensibly."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def _id_for(self, word: str) -> int:
        if word not in self.vocab:
            i = len(self.vocab)
            self.vocab[word] = i
            self.inv[i] = word
        return self.vocab[word]

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        return [self._id_for(w) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inv[i] for i in ids)


def _toy_tokenizer():
    return _WhitespaceTokenizer()


def _filler_pool():
    return [f"filler sentence number {i} about nothing in particular." for i in range(50)]


def test_needle_appears_in_document_text():
    tok = _toy_tokenizer()
    doc = build_niah_document(
        _filler_pool(), context_length_tokens=100, needle_depth_pct=50, tokenizer=tok, seed=1
    )
    assert doc.needle_fact.split()[0].lower().strip(".,") in doc.text.lower()


def test_needle_depth_zero_puts_needle_near_start():
    tok = _toy_tokenizer()
    doc = build_niah_document(
        _filler_pool(), context_length_tokens=200, needle_depth_pct=0, tokenizer=tok, seed=2
    )
    first_quarter = doc.text[: len(doc.text) // 4]
    assert doc.needle_fact.split()[0] in first_quarter or doc.needle_fact.split()[1] in first_quarter


def test_invalid_depth_raises():
    tok = _toy_tokenizer()
    import pytest
    with pytest.raises(ValueError):
        build_niah_document(_filler_pool(), context_length_tokens=50, needle_depth_pct=150, tokenizer=tok)


def test_build_suite_covers_full_grid():
    tok = _toy_tokenizer()
    docs = build_niah_suite(_filler_pool(), tokenizer=tok, lengths=(100, 200), depths_pct=(0, 50, 100))
    assert len(docs) == 2 * 3
    combos = {(d.context_length_tokens, d.needle_depth_pct) for d in docs}
    assert combos == {(100, 0), (100, 50), (100, 100), (200, 0), (200, 50), (200, 100)}


def test_score_retrieval_case_insensitive_substring():
    assert score_retrieval("I believe the answer is Zephyrine-9, based on the text.", "zephyrine-9")
    assert not score_retrieval("I don't know.", "zephyrine-9")


def test_build_accuracy_matrix_shapes():
    tok = _toy_tokenizer()
    docs = build_niah_suite(_filler_pool(), tokenizer=tok, lengths=(100,), depths_pct=(0, 100))
    answers = [docs[0].answer, "wrong answer"]  # first correct, second wrong
    matrix = build_accuracy_matrix(docs, answers)
    assert matrix[(100, 0)] is True
    assert matrix[(100, 100)] is False
