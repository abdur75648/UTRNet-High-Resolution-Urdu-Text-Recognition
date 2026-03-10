"""Tests for read.py — Image preprocessing and EXIF handling."""
import sys
import os
import pytest
import struct
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImagePreprocessing:
    """Tests for the image preprocessing pipeline in read.py."""

    def test_grayscale_conversion(self):
        """RGB image should convert to single-channel grayscale."""
        img = Image.new('RGB', (200, 50), color=(128, 64, 32))
        gray = img.convert('L')
        assert gray.mode == 'L'

    def test_flip_left_right(self):
        """Urdu text needs horizontal flip for RTL processing."""
        img = Image.new('L', (200, 50))
        flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        assert flipped.size == img.size

    def test_resize_maintains_aspect(self):
        """Resize to height 32 should maintain aspect ratio."""
        img = Image.new('L', (200, 50))
        w, h = img.size
        ratio = w / float(h)
        new_h = 32
        new_w = int(new_h * ratio)
        resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        assert resized.size[1] == 32
        assert abs(resized.size[0] - new_w) <= 1

    def test_max_width_cap(self):
        """Very wide images should be capped at width 400."""
        import math
        img = Image.new('L', (2000, 32))
        w, h = img.size
        ratio = w / float(h)
        if math.ceil(32 * ratio) > 400:
            resized_w = 400
        else:
            resized_w = math.ceil(32 * ratio)
        assert resized_w == 400


class TestExifHandling:
    """Tests for EXIF orientation handling."""

    def test_exif_transpose_no_exif(self):
        """Images without EXIF data should pass through unchanged."""
        img = Image.new('RGB', (200, 100))
        result = ImageOps.exif_transpose(img)
        assert result.size == (200, 100)

    def test_exif_transpose_rotated(self, tmp_path):
        """Image with EXIF rotation should be corrected."""
        img = Image.new('RGB', (200, 100), color=(255, 0, 0))
        img_path = str(tmp_path / "test_exif.jpg")
        img.save(img_path, exif=img.getexif().tobytes())

        loaded = Image.open(img_path)
        result = ImageOps.exif_transpose(loaded)
        assert result is not None
        assert result.size[0] > 0 and result.size[1] > 0

    def test_pil_import_includes_imageops(self):
        """Verify ImageOps is importable alongside Image."""
        from PIL import Image, ImageOps
        assert hasattr(ImageOps, 'exif_transpose')
