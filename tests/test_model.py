"""Tests for model.py — Model initialization and forward pass."""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model
from utils import CTCLabelConverter


@pytest.fixture
def model_opt():
    """Create model configuration matching UTRNet-Large defaults."""
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
    return opt


class TestModelInit:
    """Tests for model creation."""

    def test_model_creates_successfully(self, model_opt):
        """Model should initialize without errors."""
        model = Model(model_opt)
        assert model is not None

    def test_model_has_parameters(self, model_opt):
        """Model should have trainable parameters."""
        model = Model(model_opt)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0


class TestModelForward:
    """Tests for model forward pass."""

    def test_forward_pass_shape(self, model_opt):
        """Forward pass should produce correct output shape."""
        model = Model(model_opt)
        model.eval()
        dummy_input = torch.randn(1, 1, 32, 400)
        with torch.no_grad():
            output = model(dummy_input)
        # Output shape: (batch, seq_len, num_class)
        assert output.dim() == 3
        assert output.shape[0] == 1
        assert output.shape[2] == model_opt.num_class

    def test_forward_pass_batch(self, model_opt):
        """Forward pass should handle batch inputs."""
        model = Model(model_opt)
        model.eval()
        dummy_input = torch.randn(4, 1, 32, 400)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape[0] == 4


class TestModelLoad:
    """Tests for loading pretrained weights."""

    def test_load_pretrained_weights(self, model_opt):
        """Should load UTRNet-Large weights without error."""
        weights_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saved_models", "UTRNet-Large", "best_norm_ED.pth"
        )
        if not os.path.exists(weights_path):
            pytest.skip("Pre-trained weights not available")

        model = Model(model_opt)
        model.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=True)
        )
        model.eval()
        dummy_input = torch.randn(1, 1, 32, 400)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape[0] == 1
