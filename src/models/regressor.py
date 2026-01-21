from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.init as init

from .tcn import TCNBlock


class Regressor(nn.Module):
    """Pose-to-accelerometer regressor built from configurable TCN blocks.

    Supports two modes via `separate_channel`:
    - False (default): Single shared TCN backbone outputting all channels together.
      More parameter-efficient and allows learning cross-channel correlations.
    - True: Separate TCN backbone per output channel, matching the paper's approach
      "We train one regression model per sensor position and channel."
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
        separate_channel: bool = False,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.num_joints = num_joints
        self.window_length = window_length
        self.output_channels = output_channels if output_channels else in_ch
        self.separate_channel = separate_channel
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

        self.temporal_hidden_channels = temporal_hidden_channels or 16
        self.temporal_kernel_size = temporal_kernel_size or 5
        self.temporal_dilation = temporal_dilation or 1
        self.temporal_dropout = temporal_dropout if temporal_dropout is not None else 0.1

        if self.separate_channel:
            self._build_separate_channel_model(num_joint_blocks, use_batch_norm, fc_dropout)
        else:
            self._build_shared_model(num_joint_blocks, use_batch_norm, fc_dropout)

        self._initialize_weights()

    def _build_shared_model(self, num_joint_blocks: int, use_batch_norm: bool, fc_dropout: float):
        """Build single shared TCN backbone outputting all channels."""
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
        self.f_out = self.output_channels * self.window_length
        self.fc_dropout_layer = nn.Dropout(fc_dropout) if fc_dropout > 0.0 else None

        if self.fc_hidden:
            self.fc = nn.Sequential(
                nn.Linear(self.f_in, self.fc_hidden, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(fc_dropout) if fc_dropout > 0.0 else nn.Identity(),
                nn.Linear(self.fc_hidden, self.f_out, bias=True),
            )
        else:
            self.fc = nn.Linear(self.f_in, self.f_out, bias=True)

    def _build_separate_channel_model(self, num_joint_blocks: int, use_batch_norm: bool, fc_dropout: float):
        """Build separate TCN backbone per output channel."""
        self.channel_regressors = nn.ModuleList()
        for _ in range(self.output_channels):
            # Each channel gets its own joint blocks
            joint_blocks = nn.ModuleList()
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
                joint_blocks.append(block)
                in_channels = self.joint_hidden_channels[idx]

            temporal_block = TCNBlock(
                in_ch=self.joint_hidden_channels[-1] * self.num_joints,
                out_ch=self.temporal_hidden_channels,
                num_joints=1,
                kernel_size=self.temporal_kernel_size,
                dilation=self.temporal_dilation,
                dropout=self.temporal_dropout,
                use_batch_norm=use_batch_norm,
            )

            f_in = self.temporal_hidden_channels * self.window_length
            fc_dropout_layer = nn.Dropout(fc_dropout) if fc_dropout > 0.0 else None

            if self.fc_hidden:
                fc = nn.Sequential(
                    nn.Linear(f_in, self.fc_hidden, bias=True),
                    nn.LeakyReLU(),
                    nn.Dropout(fc_dropout) if fc_dropout > 0.0 else nn.Identity(),
                    nn.Linear(self.fc_hidden, self.window_length, bias=True),
                )
            else:
                fc = nn.Linear(f_in, self.window_length, bias=True)

            self.channel_regressors.append(nn.ModuleDict({
                'joint_blocks': joint_blocks,
                'temporal_block': temporal_block,
                'fc_dropout': fc_dropout_layer if fc_dropout_layer else nn.Identity(),
                'fc': fc,
            }))

        self.f_in = self.temporal_hidden_channels * self.window_length

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if self.separate_channel:
            return self._forward_separate(x)
        else:
            return self._forward_shared(x)

    def _forward_shared(self, x):
        """Forward pass with shared TCN backbone."""
        for block in self.joint_blocks:
            x = block(x)

        batch_size = x.size(0)
        x = x.reshape(batch_size, self.joint_hidden_channels[-1] * self.num_joints, 1, self.window_length)
        x = self.temporal_block(x)

        x = x.reshape(batch_size, self.f_in)
        if self.fc_dropout_layer is not None:
            x = self.fc_dropout_layer(x)
        x = self.fc(x)
        x = x.view(batch_size, self.output_channels, self.window_length)
        return x

    def _forward_separate(self, x):
        """Forward pass with separate TCN backbone per channel."""
        outputs = []
        for reg in self.channel_regressors:
            h = x
            for block in reg['joint_blocks']:
                h = block(h)

            batch_size = h.size(0)
            h = h.reshape(batch_size, self.joint_hidden_channels[-1] * self.num_joints, 1, self.window_length)
            h = reg['temporal_block'](h)

            h = h.reshape(batch_size, self.f_in)
            h = reg['fc_dropout'](h)
            h = reg['fc'](h)  # (batch, window_length)
            outputs.append(h)

        return torch.stack(outputs, dim=1)  # (batch, output_channels, window_length)
