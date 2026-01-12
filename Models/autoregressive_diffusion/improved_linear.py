"""
Improved Linear Architecture for Autoregressive Diffusion Models
Contains InvertedAttention and TemporalLinearStep components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class InvertedAttention(nn.Module):
    """
    Inverted Attention mechanism that reverses the traditional attention pattern.
    Instead of attending from query to key-value, this attends from key-value to query.
    Useful for capturing inverse relationships in temporal sequences.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for inverted attention.
        
        Args:
            query: (batch_size, seq_len, hidden_dim)
            key: (batch_size, seq_len, hidden_dim)
            value: (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            output: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.q_proj(query)  # (batch_size, seq_len, hidden_dim)
        K = self.k_proj(key)     # (batch_size, seq_len, hidden_dim)
        V = self.v_proj(value)   # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Inverted attention: compute attention from K,V to Q
        # Instead of Q @ K^T, we compute K^T @ Q
        scores = torch.matmul(K.transpose(-2, -1), Q.transpose(-2, -1))
        scores = scores * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V.transpose(-2, -1)).transpose(-2, -1)
        
        # Reshape back to original dimensions
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attention_weights


class TemporalLinearStep(nn.Module):
    """
    Temporal Linear Step module that processes sequential data with linear transformations
    while maintaining temporal coherence through residual connections and layer normalization.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: float = 4.0,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_gelu: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
        intermediate_dim = int(hidden_dim * expansion_factor)
        
        # Temporal processing layers
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.temporal_linear2 = nn.Linear(intermediate_dim, hidden_dim)
        
        # Activation function
        self.activation = nn.GELU() if use_gelu else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal linear step.
        
        Args:
            x: (batch_size, seq_len, hidden_dim)
            
        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        # Residual connection
        residual = x if self.use_residual else None
        
        # Normalize
        x = self.temporal_norm(x)
        
        # Linear transformations with activation
        x = self.temporal_linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.temporal_linear2(x)
        x = self.dropout(x)
        
        # Add residual connection
        if residual is not None:
            x = x + residual
        
        # Final normalization
        x = self.output_norm(x)
        
        return x


class ImprovedLinearBlock(nn.Module):
    """
    Improved Linear Block combining InvertedAttention and TemporalLinearStep.
    This block forms the fundamental building block of the improved architecture.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion_factor: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_residual: bool = True,
        use_gelu: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Inverted attention layer
        self.inverted_attention = InvertedAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=True
        )
        
        # Temporal linear step
        self.temporal_step = TemporalLinearStep(
            hidden_dim=hidden_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            use_residual=use_residual,
            use_gelu=use_gelu
        )
        
        # Layer normalization for attention output
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for improved linear block.
        
        Args:
            x: (batch_size, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            
        Returns:
            output: (batch_size, seq_len, hidden_dim)
            attention_weights: attention weights from InvertedAttention
        """
        # Inverted attention
        attn_output, attention_weights = self.inverted_attention(x, x, x, mask=attention_mask)
        attn_output = self.attn_norm(attn_output + x)  # Residual connection
        
        # Temporal linear step
        output = self.temporal_step(attn_output)
        
        return output, attention_weights


class ImprovedLinearModel(nn.Module):
    """
    Complete Improved Linear Model for Autoregressive Diffusion.
    Stacks multiple ImprovedLinearBlocks with input/output projections.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 6,
        num_heads: int = 8,
        expansion_factor: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_residual: bool = True,
        use_gelu: bool = True,
        max_seq_length: int = 512
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_length, hidden_dim)
        
        # Stack of improved linear blocks
        self.blocks = nn.ModuleList([
            ImprovedLinearBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_residual=use_residual,
                use_gelu=use_gelu
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def _create_positional_encoding(
        self,
        max_seq_length: int,
        hidden_dim: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        
        pe = torch.zeros(max_seq_length, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_seq_length, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass for improved linear model.
        
        Args:
            x: (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights from all blocks
            
        Returns:
            output: (batch_size, seq_len, output_dim)
            attention_weights: Optional dict of attention weights from each block
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        
        # Process through improved linear blocks
        attention_weights_list = []
        for block in self.blocks:
            x, attn_weights = block(x, attention_mask=attention_mask)
            if return_attention_weights:
                attention_weights_list.append(attn_weights)
        
        # Output projection
        x = self.output_norm(x)
        output = self.output_proj(x)  # (batch_size, seq_len, output_dim)
        
        result = {"output": output}
        
        if return_attention_weights:
            result["attention_weights"] = attention_weights_list
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "hidden_dim": self.hidden_dim,
            "max_seq_length": self.max_seq_length,
            "num_blocks": len(self.blocks)
        }


# Utility functions for model creation
def create_improved_linear_model(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_blocks: int = 6,
    num_heads: int = 8,
    **kwargs
) -> ImprovedLinearModel:
    """
    Factory function to create an ImprovedLinearModel.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension (default: 256)
        num_blocks: Number of improved linear blocks (default: 6)
        num_heads: Number of attention heads (default: 8)
        **kwargs: Additional arguments to pass to ImprovedLinearModel
        
    Returns:
        ImprovedLinearModel instance
    """
    return ImprovedLinearModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    seq_len = 128
    input_dim = 32
    output_dim = 32
    hidden_dim = 256
    
    # Create model
    model = create_improved_linear_model(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_blocks=6,
        num_heads=8
    )
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    result = model(x, return_attention_weights=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result['output'].shape}")
    print(f"Number of attention weight tensors: {len(result['attention_weights'])}")
    print(f"\nModel configuration:")
    print(model.get_config())
    
    # Print model summary
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
