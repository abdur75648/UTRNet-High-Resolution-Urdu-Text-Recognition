"""Tests for process_document.py — text_recognizer and pipeline functions."""
import sys
import os
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTextRecognizer:
    """Tests for the text_recognizer function."""

    @pytest.fixture
    def loaded_model(self):
        """Load UTRNet model for recognition tests."""
        from model import Model
        from utils import CTCLabelConverter
        glyphs_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "UrduGlyphs.txt"
        )
        with open(glyphs_path, "r", encoding="utf-8") as f:
            content = f.readlines()
        chars = ''.join([str(c).strip('\n') for c in content]) + " "
        converter = CTCLabelConverter(chars)

        class Opt:
            pass
        opt = Opt()
        opt.num_class = len(converter.character)
        opt.device = torch.device('cpu')
        opt.FeatureExtraction = 'HRNet'
        opt.SequenceModeling = 'DBiLSTM'
        opt.Prediction = 'CTC'
        opt.num_fiducial = 20
        opt.input_channel = 1
        opt.output_channel = 32
        opt.hidden_size = 256
        opt.imgH = 32
        opt.imgW = 400
        opt.batch_max_length = 100

        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saved_models", "UTRNet-Large", "best_norm_ED.pth"
        )
        if not os.path.exists(weights_path):
            pytest.skip("Pre-trained weights not available")

        model = Model(opt)
        model.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=True)
        )
        model.eval()
        return model, converter, torch.device('cpu')

    def test_recognizer_returns_string(self, loaded_model):
        """text_recognizer should return a string."""
        from process_document import text_recognizer
        model, converter, device = loaded_model
        img = Image.new('RGB', (200, 50), color=(255, 255, 255))
        result = text_recognizer(img, model, converter, device)
        assert isinstance(result, str)

    def test_recognizer_handles_small_image(self, loaded_model):
        """Should handle very small crop images."""
        from process_document import text_recognizer
        model, converter, device = loaded_model
        img = Image.new('RGB', (20, 10), color=(255, 255, 255))
        result = text_recognizer(img, model, converter, device)
        assert isinstance(result, str)

    def test_recognizer_handles_wide_image(self, loaded_model):
        """Should handle images wider than 400px."""
        from process_document import text_recognizer
        model, converter, device = loaded_model
        img = Image.new('RGB', (1000, 50), color=(255, 255, 255))
        result = text_recognizer(img, model, converter, device)
        assert isinstance(result, str)


class TestLoadRecognitionModel:
    """Tests for the model loading function."""

    def test_load_model_returns_tuple(self):
        """load_recognition_model should return (model, converter)."""
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saved_models", "UTRNet-Large", "best_norm_ED.pth"
        )
        if not os.path.exists(weights_path):
            pytest.skip("Pre-trained weights not available")

        from process_document import load_recognition_model
        model, converter = load_recognition_model(
            weights_path, torch.device('cpu')
        )
        assert model is not None
        assert converter is not None
