"""Tests for advanced layer implementations."""

import torch

from src.arc.ml.layers import (
    AddLayer,
    ConcatenateLayer,
    EmbeddingLayer,
    GRULayer,
    LayerNormLayer,
    LSTMLayer,
    MultiHeadAttentionLayer,
    PositionalEncodingLayer,
    TransformerEncoderLayerCustom,
    get_layer_class,
)


class TestAdvancedLayers:
    """Test advanced layer implementations."""

    def test_embedding_layer(self):
        """Test EmbeddingLayer implementation."""
        layer = EmbeddingLayer(num_embeddings=100, embedding_dim=32)

        # Test forward pass
        x = torch.randint(0, 100, (10, 5))  # batch_size=10, seq_len=5
        output = layer(x)

        assert output.shape == (10, 5, 32)
        assert isinstance(output, torch.Tensor)

    def test_multihead_attention_layer(self):
        """Test MultiHeadAttentionLayer implementation."""
        layer = MultiHeadAttentionLayer(embed_dim=64, num_heads=8, batch_first=True)

        # Test self-attention
        x = torch.randn(4, 10, 64)  # batch_size=4, seq_len=10, embed_dim=64
        output = layer(x)

        assert output.shape == (4, 10, 64)
        assert isinstance(output, torch.Tensor)

    def test_multihead_attention_with_explicit_inputs(self):
        """Test MultiHeadAttentionLayer with explicit query, key, value."""
        layer = MultiHeadAttentionLayer(embed_dim=64, num_heads=8, batch_first=True)

        # Test with explicit inputs
        query = torch.randn(4, 10, 64)
        key = torch.randn(4, 12, 64)
        value = torch.randn(4, 12, 64)

        output = layer(query=query, key=key, value=value)

        assert output.shape == (4, 10, 64)  # Output shape matches query
        assert isinstance(output, torch.Tensor)

    def test_transformer_encoder_layer(self):
        """Test TransformerEncoderLayerCustom implementation."""
        layer = TransformerEncoderLayerCustom(
            d_model=256, nhead=8, dim_feedforward=1024, batch_first=True
        )

        # Test forward pass
        x = torch.randn(3, 15, 256)  # batch_size=3, seq_len=15, d_model=256
        output = layer(x)

        assert output.shape == (3, 15, 256)
        assert isinstance(output, torch.Tensor)

    def test_positional_encoding_layer(self):
        """Test PositionalEncodingLayer implementation."""
        layer = PositionalEncodingLayer(d_model=128, max_len=100, dropout=0.1)

        # Test forward pass
        x = torch.randn(2, 20, 128)  # batch_size=2, seq_len=20, d_model=128
        output = layer(x)

        assert output.shape == (2, 20, 128)
        assert isinstance(output, torch.Tensor)
        # Output should be different from input (positional encoding added)
        assert not torch.equal(output, x)

    def test_layer_norm_layer(self):
        """Test LayerNormLayer implementation."""
        layer = LayerNormLayer(normalized_shape=64)

        # Test forward pass
        x = torch.randn(5, 10, 64)
        output = layer(x)

        assert output.shape == (5, 10, 64)
        assert isinstance(output, torch.Tensor)

        # Check normalization properties (approximately)
        mean = output.mean(dim=-1, keepdim=True)
        var = output.var(dim=-1, keepdim=True, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)

    def test_concatenate_layer(self):
        """Test ConcatenateLayer implementation."""
        layer = ConcatenateLayer(dim=-1)

        # Test concatenation
        x1 = torch.randn(3, 5, 10)
        x2 = torch.randn(3, 5, 8)
        x3 = torch.randn(3, 5, 12)

        output = layer(input1=x1, input2=x2, input3=x3)

        assert output.shape == (3, 5, 30)  # 10 + 8 + 12 = 30
        assert isinstance(output, torch.Tensor)

    def test_concatenate_layer_single_input_fallback(self):
        """Test ConcatenateLayer with single input (fallback behavior)."""
        layer = ConcatenateLayer(dim=-1)

        x = torch.randn(3, 5, 10)

        # Single input should fall back to pass-through behavior
        output = layer(input1=x)
        assert torch.equal(output, x)

    def test_add_layer(self):
        """Test AddLayer implementation."""
        layer = AddLayer()

        # Test element-wise addition
        x1 = torch.randn(3, 5, 10)
        x2 = torch.randn(3, 5, 10)
        x3 = torch.randn(3, 5, 10)

        output = layer(input1=x1, input2=x2, input3=x3)

        expected = x1 + x2 + x3
        assert output.shape == (3, 5, 10)
        assert torch.allclose(output, expected)

    def test_add_layer_single_input_fallback(self):
        """Test AddLayer with single input (fallback behavior)."""
        layer = AddLayer()

        x = torch.randn(3, 5, 10)

        # Single input should fall back to pass-through behavior
        output = layer(input1=x)
        assert torch.equal(output, x)

    def test_lstm_layer(self):
        """Test LSTMLayer implementation."""
        layer = LSTMLayer(input_size=32, hidden_size=64, num_layers=2, batch_first=True)

        # Test forward pass
        x = torch.randn(4, 12, 32)  # batch_size=4, seq_len=12, input_size=32
        output = layer(x)

        assert output.shape == (4, 12, 64)  # hidden_size=64
        assert isinstance(output, torch.Tensor)

    def test_gru_layer(self):
        """Test GRULayer implementation."""
        layer = GRULayer(input_size=48, hidden_size=96, num_layers=1, batch_first=True)

        # Test forward pass
        x = torch.randn(6, 8, 48)  # batch_size=6, seq_len=8, input_size=48
        output = layer(x)

        assert output.shape == (6, 8, 96)  # hidden_size=96
        assert isinstance(output, torch.Tensor)

    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM."""
        layer = LSTMLayer(
            input_size=16,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Test forward pass
        x = torch.randn(2, 10, 16)
        output = layer(x)

        # Bidirectional doubles the hidden size
        assert output.shape == (2, 10, 64)  # 32 * 2 = 64
        assert isinstance(output, torch.Tensor)


class TestAdvancedLayerRegistry:
    """Test advanced layer registry functionality."""

    def test_get_advanced_layer_classes(self):
        """Test getting advanced layer classes."""
        # Test embedding layers
        assert get_layer_class("core.Embedding") == EmbeddingLayer

        # Test attention layers
        assert get_layer_class("core.MultiHeadAttention") == MultiHeadAttentionLayer
        assert (
            get_layer_class("core.TransformerEncoderLayer")
            == TransformerEncoderLayerCustom
        )
        assert get_layer_class("core.PositionalEncoding") == PositionalEncodingLayer

        # Test normalization layers
        assert get_layer_class("core.LayerNorm") == LayerNormLayer

        # Test routing layers
        assert get_layer_class("core.Concatenate") == ConcatenateLayer
        assert get_layer_class("core.Add") == AddLayer

        # Test sequence layers
        assert get_layer_class("core.LSTM") == LSTMLayer
        assert get_layer_class("core.GRU") == GRULayer

    def test_advanced_layer_registry_completeness(self):
        """Test that registry contains all expected advanced layer types."""
        expected_advanced_layers = [
            "core.Embedding",
            "core.MultiHeadAttention",
            "core.TransformerEncoderLayer",
            "core.PositionalEncoding",
            "core.LayerNorm",
            "core.Concatenate",
            "core.Add",
            "core.LSTM",
            "core.GRU",
        ]

        from src.arc.ml.layers import LAYER_REGISTRY

        for layer_type in expected_advanced_layers:
            assert layer_type in LAYER_REGISTRY, f"Missing layer type: {layer_type}"
