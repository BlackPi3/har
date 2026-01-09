"""
Domain adaptation losses for sim-to-real IMU classification.

Contains:
- MMDLoss: Maximum Mean Discrepancy for domain alignment
- ContrastiveLoss: Class-aware contrastive loss for structure preservation

Reference: Wilson et al. (2021) CALDA framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy loss for domain alignment.

    MMD measures the distance between two distributions in a reproducing
    kernel Hilbert space (RKHS). Uses Gaussian kernel with multiple bandwidths.

    Reference: Xu et al. (2022), Wilson et al. (2020)

    Args:
        kernel_mul: Multiplier for kernel bandwidth scaling
        kernel_num: Number of kernels with different bandwidths
    """

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian kernel matrix with multiple bandwidths."""
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # Compute pairwise L2 distances
        # total: (n_samples, feat_dim)
        # We want: L2_distance[i,j] = ||total[i] - total[j]||^2
        total0 = total.unsqueeze(0)  # (1, n_samples, feat_dim)
        total1 = total.unsqueeze(1)  # (n_samples, 1, feat_dim)
        L2_distance = ((total0 - total1) ** 2).sum(2)  # (n_samples, n_samples)

        # Compute bandwidth using median heuristic
        # Avoid division by zero
        bandwidth = L2_distance.sum() / (n_samples ** 2 - n_samples + 1e-8)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))

        # Create multiple bandwidths
        bandwidth_list = [
            bandwidth * (self.kernel_mul ** i)
            for i in range(self.kernel_num)
        ]

        # Compute kernel values for each bandwidth and sum
        kernel_val = sum(
            torch.exp(-L2_distance / (bw + 1e-8))
            for bw in bandwidth_list
        )

        return kernel_val

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss between source and target features.

        Args:
            source: Source domain features (real) [batch_size, feature_dim]
            target: Target domain features (sim) [batch_size, feature_dim]

        Returns:
            MMD loss value (scalar tensor)
        """
        batch_size = source.size(0)

        # Handle mismatched batch sizes by truncating to smaller
        if source.size(0) != target.size(0):
            min_size = min(source.size(0), target.size(0))
            source = source[:min_size]
            target = target[:min_size]
            batch_size = min_size

        kernels = self.gaussian_kernel(source, target)

        # Split kernel matrix into blocks
        XX = kernels[:batch_size, :batch_size]  # source-source
        YY = kernels[batch_size:, batch_size:]  # target-target
        XY = kernels[:batch_size, batch_size:]  # source-target
        YX = kernels[batch_size:, :batch_size]  # target-source

        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)

        return loss


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for class structure preservation.

    Pulls same-class samples together and pushes different-class samples apart.
    This prevents feature collapse during domain alignment.

    Reference: Wilson et al. (2021) CALDA, Khosla et al. (2020) SupCon

    Args:
        temperature: Temperature scaling for softmax (lower = sharper)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: Feature vectors [batch_size, feature_dim]
            labels: Class labels [batch_size]

        Returns:
            Contrastive loss value (scalar tensor)
        """
        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device)

        # L2 normalize features (important for contrastive learning)
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix: sim[i,j] = features[i] · features[j] / temperature
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()  # (batch, batch)

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # For numerical stability, subtract max from logits
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log-softmax
        exp_logits = torch.exp(logits)

        # Exclude self from denominator
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True) - exp_logits.diag().view(-1, 1) + 1e-8
        )

        # Compute mean of log-likelihood over positive pairs
        # Avoid division by zero when no positive pairs exist
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()

        return loss


class CombinedDomainLoss(nn.Module):
    """
    Combined MMD + Contrastive loss for domain adaptation.

    Convenience wrapper that computes both losses and returns weighted sum.

    Args:
        lambda_mmd: Weight for MMD loss (default: 0.5)
        lambda_contrastive: Weight for contrastive loss (default: 0.3)
        mmd_kernel_mul: MMD kernel bandwidth multiplier
        mmd_kernel_num: Number of MMD kernels
        contrastive_temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        lambda_mmd: float = 0.5,
        lambda_contrastive: float = 0.3,
        mmd_kernel_mul: float = 2.0,
        mmd_kernel_num: int = 5,
        contrastive_temperature: float = 0.5,
    ):
        super().__init__()
        self.lambda_mmd = lambda_mmd
        self.lambda_contrastive = lambda_contrastive
        self.mmd_loss = MMDLoss(kernel_mul=mmd_kernel_mul, kernel_num=mmd_kernel_num)
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)

    def forward(
        self,
        real_features: torch.Tensor,
        sim_features: torch.Tensor,
        real_labels: torch.Tensor,
        sim_labels: torch.Tensor,
    ) -> tuple:
        """
        Compute combined domain adaptation loss.

        Args:
            real_features: Real domain features [batch_size, feature_dim]
            sim_features: Simulated domain features [batch_size, feature_dim]
            real_labels: Real domain class labels [batch_size]
            sim_labels: Simulated domain class labels [batch_size]

        Returns:
            Tuple of (total_loss, mmd_loss, contrastive_loss)
        """
        # MMD loss between domains
        mmd = self.mmd_loss(real_features, sim_features)

        # Contrastive loss on combined features
        all_features = torch.cat([real_features, sim_features], dim=0)
        all_labels = torch.cat([real_labels, sim_labels], dim=0)
        contrastive = self.contrastive_loss(all_features, all_labels)

        # Weighted sum
        total = self.lambda_mmd * mmd + self.lambda_contrastive * contrastive

        return total, mmd, contrastive
