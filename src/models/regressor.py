from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.init as init

from .tcn import TCNBlock


class SingleChannelRegressor(nn.Module):
    """Single-channel pose-to-accelerometer regressor (one TCN backbone + FC for 1 output).

    This matches the paper's approach: "We train one regression model per sensor
    position and channel."
    """

    def __init__(
        self,
        in_ch: int,
        num_joints: int,
        window_length: int,
        joint_hidden_channels: Optional[Sequence[int]] = None,
        joint_kernel_sizes: Optional[Sequence[int]] = None,
        joint_dilations: Optional[Sequence[int]] = None,
        joint_dropouts: Optional[Sequence[float]] = None,
        temporal_hidden_channels: Optional[int] = None,
        temporal_kernel_size: Optional[int] = None,
        temporal_dilation: Optional[int] = None,
        temporal_dropout: Optional[float] = None,
        fc_hidden: Optional[int] = None,
        fc_dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.num_joints = num_joints
        self.window_length = window_length
        self.fc_hidden = fc_hidden if fc_hidden and fc_hidden > 0 else None

        self.joint_hidden_channels = list(joint_hidden_channels or [32, 32, 32, 16])
        self.joint_kernel_sizes = list(joint_kernel_sizes or [5, 5, 5, 1])
        self.joint_dilations = list(joint_dilations or [1, 2, 4, 1])
        self.joint_dropouts = list(joint_dropouts or [0.0, 0.2, 0.2, 0.2])

        num_joint_blocks = len(self.joint_hidden_channels)
        for name, seq in (
            ("joint_kernel_sizes", self.joint_kernel_sizes),
            ("joint_dilations", self.joint_dilations),
            ("joint_dropouts", self.joint_dropouts),
        ):
            if len(seq) != num_joint_blocks:
                raise ValueError(
                    f"{name} must have the same length as joint_hidden_channels "
                    f"({len(seq)} vs {num_joint_blocks})."
                )

        self.joint_blocks = nn.ModuleList()
        in_channels = self.in_ch
        for idx in range(num_joint_blocks):
            block = TCNBlock(
                in_ch=in_channels,
                out_ch=self.joint_hidden_channels[idx],
                num_joints=self.num_joints,
                kernel_size=self.joint_kernel_sizes[idx],
                dilation=self.joint_dilations[idx],
                dropout=self.joint_dropouts[idx],
                use_batch_norm=use_batch_norm,
            )
            self.joint_blocks.append(block)
            in_channels = self.joint_hidden_channels[idx]

        self.temporal_hidden_channels = temporal_hidden_channels or 16
        self.temporal_kernel_size = temporal_kernel_size or 5
        self.temporal_dilation = temporal_dilation or 1
        self.temporal_dropout = temporal_dropout if temporal_dropout is not None else 0.1

        self.temporal_block = TCNBlock(
            in_ch=self.joint_hidden_channels[-1] * self.num_joints,
            out_ch=self.temporal_hidden_channels,
            num_joints=1,
            kernel_size=self.temporal_kernel_size,
            dilation=self.temporal_dilation,
            dropout=self.temporal_dropout,
            use_batch_norm=use_batch_norm,
        )

        self.f_in = self.temporal_hidden_channels * self.window_length
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout > 0.0 else None

        # Single FC head outputting 1 channel (window_length values)
        if self.fc_hidden:
            self.fc = nn.Sequential(
                nn.Linear(self.f_in, self.fc_hidden, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(fc_dropout) if fc_dropout > 0.0 else nn.Identity(),
                nn.Linear(self.fc_hidden, self.window_length, bias=True),
            )
        else:
            self.fc = nn.Linear(self.f_in, self.window_length, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        for block in self.joint_blocks:
            x = block(x)

        batch_size = x.size(0)
        x = x.reshape(batch_size, self.joint_hidden_channels[-1] * self.num_joints, 1, self.window_length)
        x = self.temporal_block(x)

        x = x.reshape(batch_size, self.f_in)
        if self.fc_dropout is not None:
            x = self.fc_dropout(x)

        x = self.fc(x)  # (batch, window_length)
        return x


class Regressor(nn.Module):
    """Multi-channel pose-to-accelerometer regressor with separate models per channel.

    This matches the paper's approach: "We train one regression model per sensor
    position and channel." Each output channel has its own complete TCN backbone
    and FC head.
    """

    def __init__(
        self,
        in_ch: int,
        num_joints: int,
        window_length: int,
        joint_hidden_channels: Optional[Sequence[int]] = None,
        joint_kernel_sizes: Optional[Sequence[int]] = None,
        joint_dilations: Optional[Sequence[int]] = None,
        joint_dropouts: Optional[Sequence[float]] = None,
        temporal_hidden_channels: Optional[int] = None,
        temporal_kernel_size: Optional[int] = None,
        temporal_dilation: Optional[int] = None,
        temporal_dropout: Optional[float] = None,
        fc_hidden: Optional[int] = None,
        fc_dropout: float = 0.0,
        use_batch_norm: bool = False,
        output_channels: Optional[int] = None,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.output_channels = output_channels if output_channels else in_ch
        self.window_length = window_length

        # Create separate regressor for each output channel
        self.channel_regressors = nn.ModuleList()
        for _ in range(self.output_channels):
            reg = SingleChannelRegressor(
                in_ch=in_ch,
                num_joints=num_joints,
                window_length=window_length,
                joint_hidden_channels=joint_hidden_channels,
                joint_kernel_sizes=joint_kernel_sizes,
                joint_dilations=joint_dilations,
                joint_dropouts=joint_dropouts,
                temporal_hidden_channels=temporal_hidden_channels,
                temporal_kernel_size=temporal_kernel_size,
                temporal_dilation=temporal_dilation,
                temporal_dropout=temporal_dropout,
                fc_hidden=fc_hidden,
                fc_dropout=fc_dropout,
                use_batch_norm=use_batch_norm,
            )
            self.channel_regressors.append(reg)

    def forward(self, x):
        # Each regressor outputs (batch, window_length)
        outputs = [reg(x) for reg in self.channel_regressors]
        # Stack to (batch, output_channels, window_length)
        return torch.stack(outputs, dim=1)
