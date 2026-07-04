"""Cycle 8 — global horizon-conditioned recurrent forecasters (LSTM / GRU)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class RecForecaster(nn.Module):
    def __init__(self, kind="lstm", in_dim=4, hidden=96, layers=2,
                 n_static=3, n_h=4, h_emb=8, dropout=0.15):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(in_dim, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.h_emb = nn.Embedding(n_h, h_emb)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static + h_emb, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x_seq, x_stat, h_idx):
        out, _ = self.rnn(x_seq)
        last = out[:, -1, :]
        z = torch.cat([last, x_stat, self.h_emb(h_idx)], dim=1)
        return self.head(z).squeeze(-1)

def train_model(kind, tr, va, device, logger, seed=42, max_epochs=60,
                patience=8, batch=512, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    Xs, Xt, h, yn, yr, mu, sd = tr
    Xsv, Xtv, hv, ynv, yrv, muv, sdv = va
    model = RecForecaster(kind=kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(Xt),
                                  torch.tensor(h), torch.tensor(yn)),
                    batch_size=batch, shuffle=True)
    vs, vt, vh = (torch.tensor(Xsv).to(device), torch.tensor(Xtv).to(device),
                  torch.tensor(hv).to(device))
    best, best_state, bad, curve = np.inf, None, 0, []
    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, tb, hb, yb in dl:
            xb, tb, hb, yb = (xb.to(device), tb.to(device),
                              hb.to(device), yb.to(device))
            opt.zero_grad()
            loss = lossf(model(xb, tb, hb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(vs, vt, vh).cpu().numpy() * sdv + muv
        vrmse = float(np.sqrt(np.mean((vp - yrv) ** 2)))
        curve.append(vrmse)
        logger.info(f"[{kind}] epoch {ep:02d} val_rmse={vrmse:.4f}")
        if vrmse < best - 1e-4:
            best, bad = vrmse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"[{kind}] early stop @ ep {ep} (best={best:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, curve

def predict(model, X, device, chunk=4096):
    Xs, Xt, h, mu, sd = X
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), chunk):
            p = model(torch.tensor(Xs[i:i+chunk]).to(device),
                      torch.tensor(Xt[i:i+chunk]).to(device),
                      torch.tensor(h[i:i+chunk]).to(device)).cpu().numpy()
            out.append(p)
    p = np.concatenate(out) if out else np.array([])
    return p * sd + mu
