"""LSTM classifier, weighted binary loss, and optimizer for B2."""
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Map a window of features to one logit for gold rising in five days.

    By default, two stacked LSTM layers feed the last time step into dropout,
    Linear(hidden_size, 32), ReLU, and Linear(32, 1). The logit is returned
    without a sigmoid because BCEWithLogitsLoss applies it numerically stably.
    Gradient clipping at 1.0 belongs in the B3 training loop, not this model.
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32),
                                  nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        """Return logits [batch] for input windows [batch, window, n_features]."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_loss(y_train):
    """Build BCEWithLogitsLoss from NumPy or tensor binary training labels.

    pos_weight scales positive-label loss by negatives / max(positives, 1)
    to account for class imbalance. It is a float32 tensor of shape [1].
    Pass raw logits: this loss incorporates the sigmoid numerically stably.
    """
    labels = torch.as_tensor(y_train)
    negatives = (labels == 0).sum().to(dtype=torch.float32)
    positives = (labels == 1).sum().clamp_min(1)
    pos_weight = (negatives / positives).reshape(1)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_optimizer(model, lr):
    """Return Adam with weight decay 1e-4; B3 clips gradients at 1.0 before stepping."""
    if lr <= 0:
        raise ValueError("lr must be greater than 0")
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
