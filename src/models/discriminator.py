import torch.nn as nn
import torch.nn.init as init


class Discriminator(nn.Module):
    """Discriminator model for adversarial training."""
    
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5,
                      stride=1, padding=2),  # Conv1D for temporal patterns
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5,
                      stride=1, padding=2),  # Output 1 score per time step
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze()
        return x
