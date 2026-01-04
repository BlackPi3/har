"""
Discriminator module for adversarial domain adaptation (Scenario 4).

The discriminator classifies features as "real" (from real accelerometer)
or "simulated" (from pose-to-IMU regressor).

Includes Gradient Reversal Layer (GRL) for domain-adversarial training.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) - Ganin et al., 2015
    
    Forward: identity function
    Backward: reverses gradients and scales by lambda
    
    This allows end-to-end training where the discriminator loss
    becomes an adversarial signal for the generator (feature extractor).
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradients and scale by lambda
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Wrapper module for GRL with configurable lambda.
    
    Args:
        lambda_: Gradient reversal strength. Can be:
            - Fixed float value
            - Scheduled during training (pass different values)
    """
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x, lambda_override: float = None):
        """
        Args:
            x: Input tensor
            lambda_override: Optional lambda value to override default
        """
        lambda_val = lambda_override if lambda_override is not None else self.lambda_
        return GradientReversalFunction.apply(x, lambda_val)


class FeatureDiscriminator(nn.Module):
    """
    Feature-level discriminator for domain classification.
    
    Classifies features as real (1) or simulated (0).
    
    Args:
        f_in: Input feature dimension (should match FE embedding_dim)
        hidden_units: List of hidden layer sizes
        dropout: Dropout probability
        use_grl: Whether to apply gradient reversal internally
        grl_lambda: Initial GRL lambda value
    """
    
    def __init__(
        self,
        f_in: int = 100,
        hidden_units: list = None,
        dropout: float = 0.3,
        use_grl: bool = True,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        
        if hidden_units is None:
            hidden_units = [64]
        
        self.use_grl = use_grl
        self.grl = GradientReversalLayer(grl_lambda) if use_grl else None
        self._grl_lambda = grl_lambda
        
        # Build MLP: f_in -> hidden -> ... -> 1
        layers = []
        prev_dim = f_in
        
        for h_dim in hidden_units:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        
        # Output layer: single logit for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def set_grl_lambda(self, lambda_: float):
        """Update GRL lambda (useful for scheduling)."""
        self._grl_lambda = lambda_
        if self.grl is not None:
            self.grl.lambda_ = lambda_
    
    def forward(self, features: torch.Tensor, apply_grl: bool = None, grl_lambda: float = None):
        """
        Args:
            features: (B, f_in) feature tensor from FE
            apply_grl: Override whether to apply GRL (None = use self.use_grl)
            grl_lambda: Override GRL lambda value
            
        Returns:
            logits: (B, 1) raw logits (use BCE with logits loss)
        """
        x = features
        
        # Apply GRL if configured
        should_apply = apply_grl if apply_grl is not None else self.use_grl
        if should_apply and self.grl is not None:
            lambda_val = grl_lambda if grl_lambda is not None else self._grl_lambda
            x = self.grl(x, lambda_val)
        
        return self.net(x)
    
    def forward_no_grl(self, features: torch.Tensor):
        """Forward pass without GRL (for discriminator loss computation)."""
        return self.net(features)


def compute_grl_lambda_schedule(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """
    Compute scheduled GRL lambda using the formula from DANN paper.
    
    Lambda increases from 0 to 1 over training, allowing the model to:
    1. Learn good features early (low lambda = weak adversarial signal)
    2. Enforce domain invariance later (high lambda = strong adversarial signal)
    
    Formula: lambda = 2 / (1 + exp(-gamma * p)) - 1
    where p = epoch / total_epochs
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        gamma: Controls steepness of schedule (default 10.0 from DANN)
        
    Returns:
        lambda value in [0, 1]
    """
    import math
    p = epoch / max(total_epochs - 1, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
