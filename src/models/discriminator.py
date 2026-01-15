"""
Discriminator module for adversarial domain adaptation (Scenario 4/42).

The discriminator classifies data as "real" or "simulated".

Two modes:
- Feature-level (Scenario 4): D operates on encoder features
- Signal-level (Scenario 42): D operates on raw accelerometer data

Includes Gradient Reversal Layer (GRL) for domain-adversarial training.

Improvements for stable training:
- Label smoothing: Prevents D from becoming overconfident
- Feature normalization: L2 normalize features for consistent magnitudes
- Spectral normalization: Constrains D's Lipschitz constant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        normalize_features: L2 normalize input features for stable training
        use_spectral_norm: Apply spectral normalization to constrain D
        label_smoothing: Smooth labels (e.g., 0.1 → real=0.9, sim=0.1)
    """
    
    def __init__(
        self,
        f_in: int = 100,
        hidden_units: list = None,
        dropout: float = 0.3,
        use_grl: bool = True,
        grl_lambda: float = 1.0,
        normalize_features: bool = True,
        use_spectral_norm: bool = False,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        if hidden_units is None:
            hidden_units = [64]
        
        self.use_grl = use_grl
        self.grl = GradientReversalLayer(grl_lambda) if use_grl else None
        self._grl_lambda = grl_lambda
        self.normalize_features = normalize_features
        self.label_smoothing = label_smoothing
        
        # Build MLP: f_in -> hidden -> ... -> 1
        layers = []
        prev_dim = f_in
        
        for h_dim in hidden_units:
            linear = nn.Linear(prev_dim, h_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.extend([
                linear,
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
        
        # L2 normalize features for stable training
        if self.normalize_features:
            x = F.normalize(x, p=2, dim=1)
        
        # Apply GRL if configured
        should_apply = apply_grl if apply_grl is not None else self.use_grl
        if should_apply and self.grl is not None:
            lambda_val = grl_lambda if grl_lambda is not None else self._grl_lambda
            x = self.grl(x, lambda_val)
        
        return self.net(x)
    
    def forward_no_grl(self, features: torch.Tensor):
        """Forward pass without GRL (for discriminator loss computation)."""
        x = features
        if self.normalize_features:
            x = F.normalize(x, p=2, dim=1)
        return self.net(x)
    
    def get_smooth_labels(self, batch_size: int, real: bool, device: torch.device):
        """
        Get smoothed labels for discriminator training.
        
        With label_smoothing=0.1:
            real: 0.9 instead of 1.0
            sim:  0.1 instead of 0.0
            
        This prevents D from becoming overconfident and maintains gradient signal.
        """
        if real:
            return torch.full((batch_size, 1), 1.0 - self.label_smoothing, device=device)
        else:
            return torch.full((batch_size, 1), self.label_smoothing, device=device)


class SignalDiscriminator(nn.Module):
    """
    Signal-level discriminator for raw accelerometer data (Scenario 42).

    Uses 1D convolutions to process temporal signal, then classifies as
    real (from sensor) or simulated (from pose2imu regressor).

    This directly trains the regressor to produce realistic accelerometer signals,
    rather than relying on the encoder to align features.

    Args:
        n_channels: Number of input channels (default 3 for accelerometer)
        window_size: Temporal window length
        hidden_channels: List of conv layer channel sizes
        dropout: Dropout probability
        use_grl: Whether to apply gradient reversal internally
        grl_lambda: Initial GRL lambda value
        use_spectral_norm: Apply spectral normalization to constrain D
        label_smoothing: Smooth labels (e.g., 0.1 → real=0.9, sim=0.1)
    """

    def __init__(
        self,
        n_channels: int = 3,
        window_size: int = 100,
        hidden_channels: list = None,
        dropout: float = 0.3,
        use_grl: bool = True,
        grl_lambda: float = 1.0,
        use_spectral_norm: bool = False,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64]

        self.use_grl = use_grl
        self.grl = GradientReversalLayer(grl_lambda) if use_grl else None
        self._grl_lambda = grl_lambda
        self.label_smoothing = label_smoothing
        self.n_channels = n_channels
        self.window_size = window_size

        # Build 1D CNN: (B, n_channels, window_size) -> (B, hidden, reduced_size)
        conv_layers = []
        prev_ch = n_channels

        for h_ch in hidden_channels:
            conv = nn.Conv1d(prev_ch, h_ch, kernel_size=5, stride=2, padding=2)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            conv_layers.extend([
                conv,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev_ch = h_ch

        self.conv_net = nn.Sequential(*conv_layers)

        # Compute flattened size after convolutions
        test_input = torch.zeros(1, n_channels, window_size)
        with torch.no_grad():
            test_output = self.conv_net(test_input)
            flat_size = test_output.view(1, -1).size(1)

        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_grl_lambda(self, lambda_: float):
        """Update GRL lambda (useful for scheduling)."""
        self._grl_lambda = lambda_
        if self.grl is not None:
            self.grl.lambda_ = lambda_

    def forward(self, signal: torch.Tensor, apply_grl: bool = None, grl_lambda: float = None):
        """
        Args:
            signal: (B, n_channels, window_size) accelerometer tensor
            apply_grl: Override whether to apply GRL (None = use self.use_grl)
            grl_lambda: Override GRL lambda value

        Returns:
            logits: (B, 1) raw logits (use BCE with logits loss)
        """
        x = signal

        # Apply GRL if configured (before conv for gradient reversal to regressor)
        should_apply = apply_grl if apply_grl is not None else self.use_grl
        if should_apply and self.grl is not None:
            lambda_val = grl_lambda if grl_lambda is not None else self._grl_lambda
            x = self.grl(x, lambda_val)

        # Conv feature extraction
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)

    def get_smooth_labels(self, batch_size: int, real: bool, device: torch.device):
        """Get smoothed labels for discriminator training."""
        if real:
            return torch.full((batch_size, 1), 1.0 - self.label_smoothing, device=device)
        else:
            return torch.full((batch_size, 1), self.label_smoothing, device=device)

    def forward_no_grl(self, signal: torch.Tensor):
        """Forward pass without GRL (for gradient penalty computation in WGAN)."""
        x = self.conv_net(signal)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ACSignalDiscriminator(nn.Module):
    """
    ACGAN-style Signal Discriminator for class-conditional adversarial training.

    Extends SignalDiscriminator with:
    1. Class conditioning: D evaluates "is this a real sample OF CLASS Y?"
    2. Auxiliary classifier: D also predicts activity class

    This encourages class-specific realism - the generator must produce signals
    that look like real samples of the correct activity class.

    Reference: Odena et al., "Conditional Image Synthesis with Auxiliary Classifier GANs"

    Args:
        n_channels: Number of input channels (default 3 for accelerometer)
        window_size: Temporal window length
        n_classes: Number of activity classes
        hidden_channels: List of conv layer channel sizes
        embed_dim: Dimension of class label embedding
        dropout: Dropout probability
        use_grl: Whether to apply gradient reversal internally
        grl_lambda: Initial GRL lambda value
        use_spectral_norm: Apply spectral normalization to constrain D
        label_smoothing: Smooth labels for real/fake (not used for aux classifier)
        aux_weight: Weight for auxiliary classification loss (relative to real/fake)
    """

    def __init__(
        self,
        n_channels: int = 3,
        window_size: int = 100,
        n_classes: int = 21,
        hidden_channels: list = None,
        embed_dim: int = 32,
        dropout: float = 0.3,
        use_grl: bool = True,
        grl_lambda: float = 1.0,
        use_spectral_norm: bool = False,
        label_smoothing: float = 0.1,
        aux_weight: float = 1.0,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64]

        self.use_grl = use_grl
        self.grl = GradientReversalLayer(grl_lambda) if use_grl else None
        self._grl_lambda = grl_lambda
        self.label_smoothing = label_smoothing
        self.n_channels = n_channels
        self.window_size = window_size
        self.n_classes = n_classes
        self.aux_weight = aux_weight
        self.embed_dim = embed_dim

        # Class embedding for conditioning
        # Broadcasts across time dimension via FiLM-style conditioning
        self.class_embed = nn.Embedding(n_classes, embed_dim)

        # Build 1D CNN: (B, n_channels, window_size) -> (B, hidden, reduced_size)
        conv_layers = []
        prev_ch = n_channels

        for i, h_ch in enumerate(hidden_channels):
            conv = nn.Conv1d(prev_ch, h_ch, kernel_size=5, stride=2, padding=2)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            conv_layers.extend([
                conv,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev_ch = h_ch

        self.conv_net = nn.Sequential(*conv_layers)

        # Compute flattened size after convolutions
        test_input = torch.zeros(1, n_channels, window_size)
        with torch.no_grad():
            test_output = self.conv_net(test_input)
            self._conv_out_size = test_output.size()  # (1, C, L)
            flat_size = test_output.view(1, -1).size(1)

        # Shared representation layer (combines conv features + class embedding)
        # Class embedding is projected and concatenated
        self.fc_shared = nn.Sequential(
            nn.Linear(flat_size + embed_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )

        # Real/Fake head (discriminator)
        self.fc_adv = nn.Linear(128, 1)

        # Auxiliary classifier head (predicts activity class)
        self.fc_aux = nn.Linear(128, n_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def set_grl_lambda(self, lambda_: float):
        """Update GRL lambda (useful for scheduling)."""
        self._grl_lambda = lambda_
        if self.grl is not None:
            self.grl.lambda_ = lambda_

    def forward(
        self,
        signal: torch.Tensor,
        labels: torch.Tensor = None,
        apply_grl: bool = None,
        grl_lambda: float = None,
    ):
        """
        Forward pass with optional class conditioning.

        Args:
            signal: (B, n_channels, window_size) accelerometer tensor
            labels: (B,) class labels for conditioning (optional)
            apply_grl: Override whether to apply GRL (None = use self.use_grl)
            grl_lambda: Override GRL lambda value

        Returns:
            If labels provided:
                adv_logits: (B, 1) real/fake logits
                aux_logits: (B, n_classes) activity class logits
            If labels not provided (unconditional mode):
                adv_logits: (B, 1) real/fake logits
                aux_logits: (B, n_classes) activity class logits
        """
        x = signal

        # Apply GRL if configured (before conv for gradient reversal to regressor)
        should_apply = apply_grl if apply_grl is not None else self.use_grl
        if should_apply and self.grl is not None:
            lambda_val = grl_lambda if grl_lambda is not None else self._grl_lambda
            x = self.grl(x, lambda_val)

        # Conv feature extraction
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # (B, flat_size)

        # Class conditioning
        if labels is not None:
            class_emb = self.class_embed(labels)  # (B, embed_dim)
        else:
            # If no labels, use zero embedding (unconditional)
            class_emb = torch.zeros(x.size(0), self.embed_dim, device=x.device)

        # Concatenate features with class embedding
        x = torch.cat([x, class_emb], dim=1)  # (B, flat_size + embed_dim)

        # Shared representation
        shared = self.fc_shared(x)  # (B, 128)

        # Two heads
        adv_logits = self.fc_adv(shared)   # (B, 1) - real/fake
        aux_logits = self.fc_aux(shared)   # (B, n_classes) - activity class

        return adv_logits, aux_logits

    def forward_no_grl(self, signal: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass without GRL (for gradient penalty computation in WGAN)."""
        x = self.conv_net(signal)
        x = x.view(x.size(0), -1)

        if labels is not None:
            class_emb = self.class_embed(labels)
        else:
            class_emb = torch.zeros(x.size(0), self.embed_dim, device=x.device)

        x = torch.cat([x, class_emb], dim=1)
        shared = self.fc_shared(x)

        adv_logits = self.fc_adv(shared)
        aux_logits = self.fc_aux(shared)

        return adv_logits, aux_logits

    def get_smooth_labels(self, batch_size: int, real: bool, device: torch.device):
        """Get smoothed labels for discriminator training (real/fake head only)."""
        if real:
            return torch.full((batch_size, 1), 1.0 - self.label_smoothing, device=device)
        else:
            return torch.full((batch_size, 1), self.label_smoothing, device=device)


def compute_gradient_penalty_acgan(
    discriminator: "ACSignalDiscriminator",
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP with ACGAN discriminator.

    Same as compute_gradient_penalty but passes labels through.

    Args:
        discriminator: ACSignalDiscriminator instance
        real_samples: Real data samples (B, C, L)
        fake_samples: Generated/simulated samples, same shape as real
        labels: Class labels (B,)
        device: Torch device
        lambda_gp: Gradient penalty coefficient

    Returns:
        Gradient penalty loss term
    """
    batch_size = real_samples.size(0)

    # Random interpolation coefficient
    alpha = torch.rand(batch_size, *([1] * (real_samples.dim() - 1)), device=device)

    # Interpolate between real and fake
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Get discriminator output for interpolated samples (without GRL)
    adv_logits, _ = discriminator.forward_no_grl(interpolated, labels)

    # Compute gradients w.r.t. interpolated samples
    gradients = torch.autograd.grad(
        outputs=adv_logits,
        inputs=interpolated,
        grad_outputs=torch.ones_like(adv_logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradients and compute L2 norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Penalty: (||grad|| - 1)^2
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


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


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.

    The gradient penalty enforces the Lipschitz constraint by penalizing
    gradients that deviate from unit norm on interpolated samples.

    Formula: GP = E[(||∇D(x_interp)||_2 - 1)^2]

    Args:
        discriminator: The discriminator network
        real_samples: Real data samples (B, C, L) for signal or (B, F) for features
        fake_samples: Generated/simulated samples, same shape as real
        device: Torch device
        lambda_gp: Gradient penalty coefficient (default 10.0 from WGAN-GP paper)

    Returns:
        Gradient penalty loss term
    """
    batch_size = real_samples.size(0)

    # Random interpolation coefficient
    alpha = torch.rand(batch_size, *([1] * (real_samples.dim() - 1)), device=device)

    # Interpolate between real and fake
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Get discriminator output for interpolated samples (without GRL)
    if hasattr(discriminator, 'forward_no_grl'):
        d_interpolated = discriminator.forward_no_grl(interpolated)
    else:
        d_interpolated = discriminator(interpolated, apply_grl=False)

    # Compute gradients w.r.t. interpolated samples
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradients and compute L2 norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Penalty: (||grad|| - 1)^2
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


class WGANLoss:
    """
    Wasserstein GAN loss computation for adversarial training.

    WGAN uses the Wasserstein distance (Earth Mover's distance) instead of
    JS divergence, providing better gradient signal especially when
    distributions don't overlap well.

    Loss formulas:
    - D loss: E[D(fake)] - E[D(real)] + λ_gp * GP  (D wants to maximize real - fake)
    - G loss: -E[D(fake)]  (G wants to maximize D(fake), i.e., fool D)

    Note: With GRL, we use a single forward pass where:
    - D receives normal gradients (learns to discriminate)
    - G/FE/Regressor receive reversed gradients (learns to fool D)

    The combined loss becomes: E[D(real)] - E[D(fake)] + GP
    With GRL on fake path, this naturally creates the adversarial dynamic.
    """

    def __init__(self, lambda_gp: float = 10.0, use_gp: bool = True):
        """
        Args:
            lambda_gp: Gradient penalty coefficient
            use_gp: Whether to use gradient penalty (recommended for stability)
        """
        self.lambda_gp = lambda_gp
        self.use_gp = use_gp

    def discriminator_loss(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
        gradient_penalty: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute discriminator loss for WGAN.

        D wants: D(real) high, D(fake) low
        Loss = E[D(fake)] - E[D(real)] + GP

        Args:
            d_real: Discriminator output for real samples (B, 1)
            d_fake: Discriminator output for fake samples (B, 1)
            gradient_penalty: Pre-computed gradient penalty (optional)

        Returns:
            Discriminator loss
        """
        loss = d_fake.mean() - d_real.mean()
        if gradient_penalty is not None:
            loss = loss + gradient_penalty
        return loss

    def generator_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss for WGAN.

        G wants: D(fake) high (fool D into thinking fake is real)
        Loss = -E[D(fake)]

        Args:
            d_fake: Discriminator output for fake/generated samples

        Returns:
            Generator loss
        """
        return -d_fake.mean()

    def combined_loss_with_grl(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
        gradient_penalty: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute combined loss when using GRL (single backward pass).

        With GRL applied to fake samples before D:
        - D sees: real (normal grad) and fake (reversed grad to upstream)
        - Loss: E[D(real)] - E[D(fake)] + GP

        The GRL automatically handles the adversarial gradient reversal,
        so we optimize this single loss and gradients flow correctly:
        - To D: normal gradients (D learns to discriminate)
        - To G/FE: reversed gradients (G learns to fool D)

        Args:
            d_real: D output for real samples
            d_fake: D output for fake samples (GRL already applied in forward)
            gradient_penalty: Gradient penalty term

        Returns:
            Combined loss for single backward pass
        """
        # Wasserstein distance: D(real) - D(fake)
        # We want to maximize this for D, minimize for G
        # With GRL on fake path, minimizing this loss achieves both
        loss = d_real.mean() - d_fake.mean()
        if gradient_penalty is not None:
            loss = loss + gradient_penalty
        return loss
