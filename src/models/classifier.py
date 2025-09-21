import torch.nn as nn
import torch.nn.init as init
import math


class FeatureExtractor(nn.Module):
    """Feature extractor using 1D convolutions for temporal data."""
    
    def __init__(self, n_filters=9, filter_size=5, n_dense=100, n_channels=3, 
                 window_size=300, drop_prob=0.2, pool_filter_size=2):
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
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=filter_size,
                      padding=filter_size//2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_filters),
            nn.Dropout(p=0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=filter_size,
                      padding=filter_size//2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(n_filters),
            nn.Dropout(p=0.3)
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
    """Simple linear classifier for activity recognition."""
    
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
