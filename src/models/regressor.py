import torch.nn as nn
import torch.nn.init as init
from .tcn import TCNBlock

class Regressor(nn.Module):
    """Original regressor model with additional TCN layer."""
    
    def __init__(self, in_ch, num_joints, window_length):
        """
        Args:
            in_ch: Number of input channels
            num_joints: Number of joints (e.g., wrist, elbow, shoulder)
            window_length: Length of the input window
        """
        super(Regressor, self).__init__()

        self.in_ch = in_ch
        self.num_joints = num_joints
        self.window_length = window_length

        self.tcn1 = TCNBlock(
            in_ch=self.in_ch,
            out_ch=32,
            num_joints=self.num_joints,
            kernel_size=5,
            dilation=1,
            dropout=0,
        )

        self.tcn2 = TCNBlock(
            in_ch=32,
            out_ch=32,
            num_joints=self.num_joints,
            kernel_size=5,
            dilation=2,
            dropout=0.2,
        )

        self.tcn3 = TCNBlock(
            in_ch=32,
            out_ch=32,
            num_joints=self.num_joints,
            kernel_size=5,
            dilation=4,
            dropout=0.2,
        )

        self.tcn4 = TCNBlock(
            in_ch=32,
            out_ch=16,
            num_joints=self.num_joints,
            kernel_size=1,
            dilation=1,
            dropout=0.2,
        )

        self.tcn5 = TCNBlock(
            in_ch=16 * self.num_joints,
            out_ch=16,
            num_joints=1,
            kernel_size=5,
            dilation=1,
            dropout=0.1,
        )

        self.f_in = 16 * 1 * self.window_length
        self.f_out = self.in_ch * self.window_length
        self.fc = nn.Linear(in_features=self.f_in,
                            out_features=self.f_out, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = self.tcn4(x)

        x = x.view(-1, 16 * self.num_joints, 1, self.window_length)

        x = self.tcn5(x)  # (batch_size, 16, 1, window)

        x = x.view(-1, self.f_in)
        x = self.fc(x)
        x = x.view(-1, self.in_ch, self.window_length)

        return x
