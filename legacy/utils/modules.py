import torch.nn as nn
import torch.nn.init as init
import math
import torch

# >>> REGRESSOR models <<< #


class TCNBlock(nn.Module):
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
                # init.kaiming_normal_(
                #     m.weight, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        short_cut = self.conv0(x)

        x = self.leakyrelu(self.dropout1(self.conv1(x)))
        x = self.leakyrelu(self.dropout2(self.conv2(x)))

        return x + short_cut


class RegressorNew(nn.Module):
    def __init__(self, in_ch=3, num_joints=3, window_length=100):
        """
        input_size: the number of joints (wrist, elbow, shoulder)

        """
        super(RegressorNew, self).__init__()

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

        self.f_in = (
            16 * 3 * self.window_length
        )  # NOTE '1'. it's because we'll reshape x.
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

        x = x.view(-1, self.f_in)
        x = self.fc(x)
        x = x.view(-1, self.in_ch, self.window_length)

        return x


class Regressor(nn.Module):
    def __init__(self, in_ch, num_joints, window_length):
        """
        input_size: the number of joints (wrist, elbow, shoulder)

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
            num_joints=1,  # NOTE
            kernel_size=5,
            dilation=1,
            dropout=0.1,
        )

        self.f_in = (
            16 * 1 * self.window_length
        )  # NOTE '1'. it's because we'll reshape x.
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

        x = x.view(-1, 16 * self.num_joints, 1, self.window_length)  # NOTE

        x = self.tcn5(x)  # (batch_size, 16, 1, window)

        # print(self.f_in)
        x = x.view(-1, self.f_in)
        x = self.fc(x)
        x = x.view(-1, self.in_ch, self.window_length)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, n_filters=9, filter_size=5, n_dense=100, n_channels=3, window_size=300, drop_prob=0.2, pool_filter_size=2):
        super(FeatureExtractor, self).__init__()

        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_dense = n_dense
        self.n_channels = n_channels
        self.window_size = window_size
        self.pool_filter_size = pool_filter_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size=filter_size,
                      padding=filter_size//2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_filters),
            nn.Dropout(p=0.3)
            # nn.Dropout(p=drop_prob)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=filter_size,
                      padding=filter_size//2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_filters),
            nn.Dropout(p=0.3)
            # nn.Dropout(p=drop_prob)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=filter_size,
                      padding=filter_size//2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_filters),
            nn.Dropout(p=0.3)
            # nn.Dropout(p=drop_prob)
        )

        self.maxpool = nn.MaxPool1d(pool_filter_size, pool_filter_size)

        flatten_size = n_filters * \
            math.floor(window_size/(pool_filter_size*pool_filter_size))
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, n_dense),
            nn.LeakyReLU()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            # or isinstance(m, nn.ConvTranspose2d):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        batch_size, C, L = out.size()
        flatten = out.view(-1, C*L)
        out = self.fc(flatten)

        return out


class ActivityClassifier(nn.Module):
    def __init__(self, f_in, n_classes):
        super(ActivityClassifier, self).__init__()
        self.classfier = nn.Linear(f_in, n_classes, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.classfier(x)
        return out


class Discriminator(nn.Module):
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
        # Flatten the input from (batch size, 3, window) to (batch size, 3 * window)
        # x = x.view(x.size(0), -1)
        x = self.model(x)
        x = x.squeeze()
        return x
