"""Tests for dataset.py — NormalizePAD and related transforms."""
import sys
import os
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import NormalizePAD


class TestNormalizePAD:
    """Tests for the NormalizePAD transform."""

    def test_output_shape_single_channel(self):
        """Output should match the specified (C, H, W) dimensions."""
        transform = NormalizePAD((1, 32, 400))
        img = Image.new('L', (100, 32))
        result = transform(img)
        assert result.shape == (1, 32, 400)

    def test_output_is_tensor(self):
        """Transform should return a torch Tensor."""
        transform = NormalizePAD((1, 32, 400))
        img = Image.new('L', (100, 32))
        result = transform(img)
        assert isinstance(result, torch.Tensor)

    def test_exact_width_image(self):
        """Images at exactly max_width should work without padding."""
        transform = NormalizePAD((1, 32, 400))
        img = Image.new('L', (400, 32))
        result = transform(img)
        assert result.shape == (1, 32, 400)

    def test_narrow_image_padded(self):
        """Images narrower than max_width should be padded."""
        transform = NormalizePAD((1, 32, 400))
        img = Image.new('L', (50, 32))
        result = transform(img)
        assert result.shape == (1, 32, 400)

    def test_values_normalized(self):
        """Output values should be normalized (not raw 0-255)."""
        transform = NormalizePAD((1, 32, 400))
        img = Image.new('L', (100, 32), color=128)
        result = transform(img)
        assert result.max().item() <= 2.0  # roughly normalized range
        assert result.min().item() >= -2.0

    def test_three_channel(self):
        """Transform should work with 3-channel (RGB) images."""
        transform = NormalizePAD((3, 32, 400))
        img = Image.new('RGB', (100, 32))
        result = transform(img)
        assert result.shape == (3, 32, 400)


class TestItertools:
    """Test that itertools.accumulate works as a drop-in for torch._utils._accumulate."""

    def test_accumulate_basic(self):
        """accumulate should produce running sum."""
        from itertools import accumulate as _accumulate
        result = list(_accumulate([1, 2, 3]))
        assert result == [1, 3, 6]

    def test_accumulate_empty(self):
        """accumulate on empty list should return empty."""
        from itertools import accumulate as _accumulate
        result = list(_accumulate([]))
        assert result == []
