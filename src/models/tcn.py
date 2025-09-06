import torch.nn as nn
import torch.nn.init as init


class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block with residual connections."""
    
    def __init__(self, in_ch, out_ch, num_joints, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        self.num_joints = num_joints
        if kernel_size == 1:
            self.num_joints = 1

        self.conv0 = nn.Conv2d(in_ch, out_ch, (1, 1), padding="same")

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            (self.num_joints, kernel_size),
            padding="same",
            dilation=dilation,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            (self.num_joints, kernel_size),
            padding="same",
            dilation=dilation,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        short_cut = self.conv0(x)

        x = self.leakyrelu(self.dropout1(self.conv1(x)))
        x = self.leakyrelu(self.dropout2(self.conv2(x)))

        return x + short_cut