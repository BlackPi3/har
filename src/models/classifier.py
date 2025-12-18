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


class MmfitEncoder1D(nn.Module):
    """
    Encoder-only variant inspired by MMFit ConvAutoencoder.
    Uses strided convolutions plus a max-pool, then flattens to a dense embedding.
    """

    def __init__(
        self,
        input_channels: int = 3,
        layers: int = 3,
        kernel_size: int = 11,
        kernel_stride: int = 2,
        grouped: list | None = None,
        base_filters: list | None = None,
        pool_kernel: int = 2,
        embedding_dim: int = 100,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.layers = max(1, int(layers))
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.grouped = grouped or [1, 1, 1]
        self.base_filters = base_filters or [9, 15, 24]
        self.pool_kernel = pool_kernel
        self.embedding_dim = embedding_dim
        self.use_batch_norm = bool(use_batch_norm)

        padding = (kernel_size - 1) // 2
        g1, g2, g3 = (self.grouped + [1, 1, 1])[:3]
        f1, f2, f3 = (self.base_filters + [self.base_filters[-1]] * 3)[:3]

        convs = []
        norms = []
        convs.append(nn.Conv1d(input_channels, f1, kernel_size=kernel_size, stride=kernel_stride, padding=padding, groups=g1))
        norms.append(nn.BatchNorm1d(f1) if self.use_batch_norm else nn.Identity())
        if self.layers > 1:
            convs.append(nn.Conv1d(f1, f2, kernel_size=kernel_size, stride=kernel_stride, padding=padding, groups=g2))
            norms.append(nn.BatchNorm1d(f2) if self.use_batch_norm else nn.Identity())
        if self.layers > 2:
            convs.append(nn.Conv1d(f2, f3, kernel_size=kernel_size, stride=kernel_stride, padding=padding, groups=g3))
            norms.append(nn.BatchNorm1d(f3) if self.use_batch_norm else nn.Identity())
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel, stride=self.pool_kernel)
        # Initialize convs similarly to MMFit (default init OK, keep simple)

        self.relu = nn.ReLU()
        self.fc = None  # initialized lazily after forward shape known if embedding_dim set
        self.embedding_dim = embedding_dim

    def forward(self, x):
        out = x
        for conv, norm in zip(self.convs, self.norms):
            out = conv(out)
            out = norm(out)
            out = self.relu(out)
        out = self.pool(out)
        b, c, l = out.shape
        flat = out.view(b, c * l)
        if self.fc is None:
            self.fc = nn.Linear(flat.shape[1], self.embedding_dim)
            init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
            if self.fc.bias is not None:
                init.constant_(self.fc.bias, 0)
            self.fc = self.fc.to(out.device)
        emb = self.fc(flat)
        return emb


class MlpClassifier(nn.Module):
    """
    MLP classifier similar to MMFit multimodal FC classifier (2-3 layers with dropout).
    """

    def __init__(self, f_in: int, n_classes: int, hidden_units=None, dropout: float = 0.0):
        super().__init__()
        hidden_units = hidden_units if hidden_units is not None else [100]
        if isinstance(hidden_units, (int, float)):
            hidden_units = [int(hidden_units)]
        layers = []
        prev = f_in
        for h in hidden_units:
            layers.append(nn.Linear(prev, h, bias=True))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes, bias=True))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
