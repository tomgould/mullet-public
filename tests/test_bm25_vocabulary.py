"""Tests for mullet.knowledge.engine.bm25_vocabulary."""
import pytest
from mullet.knowledge.engine.bm25_vocabulary import load_bm25_vocabulary, save_bm25_vocabulary


def test_load_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        load_bm25_vocabulary("some_file.npz")


def test_save_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        save_bm25_vocabulary({}, "some_file.npz")
