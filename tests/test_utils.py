"""Tests for utils.py — CTCLabelConverter and AttnLabelConverter."""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CTCLabelConverter, AttnLabelConverter


@pytest.fixture
def urdu_chars():
    """Load the Urdu character set used by the model."""
    glyphs_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "UrduGlyphs.txt"
    )
    with open(glyphs_path, "r", encoding="utf-8") as f:
        content = f.readlines()
    chars = ''.join([str(c).strip('\n') for c in content]) + " "
    return chars


@pytest.fixture
def ctc_converter(urdu_chars):
    return CTCLabelConverter(urdu_chars)


@pytest.fixture
def attn_converter(urdu_chars):
    return AttnLabelConverter(urdu_chars)


class TestCTCLabelConverter:
    """Tests for CTC label encoding and decoding."""

    def test_character_set_loaded(self, ctc_converter):
        """Converter should have a non-empty character list."""
        assert len(ctc_converter.character) > 0

    def test_encode_returns_tensors(self, ctc_converter):
        """Encoding should return (text_tensor, length_tensor)."""
        text, length = ctc_converter.encode(["سلام"])
        assert isinstance(text, torch.Tensor)
        assert isinstance(length, torch.Tensor)

    def test_encode_single_char(self, ctc_converter):
        """Encoding a single character should produce length 1."""
        _, length = ctc_converter.encode(["ا"])
        assert length.item() >= 1

    def test_encode_empty_string(self, ctc_converter):
        """Encoding an empty string should produce length 0."""
        _, length = ctc_converter.encode([""])
        assert length.item() == 0

    def test_decode_returns_strings(self, ctc_converter):
        """Decoding should return a list of strings."""
        # Create a fake prediction of shape (batch, seq_len)
        seq_len = 10
        batch_size = 1
        num_chars = len(ctc_converter.character)
        preds_index = torch.randint(0, num_chars, (batch_size, seq_len))
        preds_size = torch.IntTensor([seq_len] * batch_size)
        result = ctc_converter.decode(preds_index, preds_size)
        assert isinstance(result, list)
        assert len(result) == batch_size
        assert isinstance(result[0], str)

    def test_encode_batch(self, ctc_converter):
        """Encoding a batch of texts should handle multiple strings."""
        texts = ["سلام", "پاکستان"]
        text, length = ctc_converter.encode(texts)
        assert length.shape[0] == 2

    def test_space_character_included(self, ctc_converter):
        """Space character should be in the character set."""
        assert " " in ctc_converter.character


class TestAttnLabelConverter:
    """Tests for Attention label encoding and decoding."""

    def test_character_set_has_special_tokens(self, attn_converter):
        """Attention converter should include [GO] and [s] tokens."""
        assert '[GO]' in attn_converter.character
        assert '[s]' in attn_converter.character

    def test_encode_returns_tensors(self, attn_converter):
        """Encoding should return (text_tensor, length_tensor)."""
        text, length = attn_converter.encode(["سلام"], batch_max_length=100)
        assert isinstance(text, torch.Tensor)
        assert isinstance(length, torch.Tensor)

    def test_decode_returns_strings(self, attn_converter):
        """Decoding should return a list of strings."""
        text, length = attn_converter.encode(["سلام"], batch_max_length=100)
        result = attn_converter.decode(text, length)
        assert isinstance(result, list)
        assert len(result) > 0
