


from __future__ import annotations

import math

import torch
import torch.nn as nn


class MaskedPool(nn.Module):


    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(h.dtype)
        s = (h * m).sum(1)
        n = m.sum(1).clamp(min=1.0)
        return s / n


class SeqLSTM(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, layers: int = 1, n_out: int = 2, drop: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True,
                           dropout=drop if layers > 1 else 0.0)
        self.pool = MaskedPool()
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, n_out))

    def forward(self, x, mask):
        h, _ = self.rnn(x)
        return self.head(self.pool(h, mask))


class _Chomp1d(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n].contiguous() if self.n else x


class _TCNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, d: int, drop: float):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=d)
        self.chomp1 = _Chomp1d(pad)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=d)
        self.chomp2 = _Chomp1d(pad)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        o = self.drop(self.act(self.chomp1(self.conv1(x))))
        o = self.drop(self.act(self.chomp2(self.conv2(o))))
        return self.act(o + self.down(x))


class SeqTCN(nn.Module):
    def __init__(self, n_feat: int, ch: int = 32, n_out: int = 2, drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            _TCNBlock(n_feat, ch, 3, 1, drop),
            _TCNBlock(ch, ch * 2, 3, 2, drop),
            _TCNBlock(ch * 2, ch * 2, 3, 4, drop),
        )
        self.pool = MaskedPool()
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(ch * 2, n_out))

    def forward(self, x, mask):
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.pool(h, mask))


class SeqTransformer(nn.Module):
    def __init__(self, n_feat: int, d: int = 64, nhead: int = 4, layers: int = 2,
                 max_len: int = 64, n_out: int = 2, drop: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=d * 2, dropout=drop,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.pool = MaskedPool()
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, n_out))

    def forward(self, x, mask):
        h = self.proj(x) + self.pos[:, : x.shape[1]]

        h = self.enc(h, src_key_padding_mask=~mask)
        return self.head(self.pool(h, mask))


SEQ_MODELS = {
    "lstm": lambda f, T: SeqLSTM(f),
    "tcn": lambda f, T: SeqTCN(f),
    "transformer": lambda f, T: SeqTransformer(f, max_len=max(T, 64)),
}


def build_seq_model(name: str, n_feat: int, T: int) -> nn.Module:
    if name not in SEQ_MODELS:
        raise ValueError(f"Unknown sequence model {name!r}; choices {list(SEQ_MODELS)}")
    return SEQ_MODELS[name](n_feat, T)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def pool_table(X, mask):


    import numpy as np
    m = mask[..., None].astype(np.float32)
    n = np.maximum(m.sum(1), 1.0)
    mean = (X * m).sum(1) / n
    var = (((X - mean[:, None, :]) ** 2) * m).sum(1) / n
    std = np.sqrt(np.maximum(var, 0.0))
    Xm = np.where(mask[..., None], X, -np.inf)
    mx = Xm.max(1)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return np.concatenate([mean, std, mx], axis=1)
