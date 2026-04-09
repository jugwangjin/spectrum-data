"""
DEPRECATED — not imported by `au_region_analysis.py`.

보관용: PyTorch가 있는 환경에서만 `train_spectrum_softmax_ae` 등을 실험적으로 사용.
메인 파이프라인은 torch 없이 동작하도록 AE 경로를 제거했다.

Spectral autoencoder: MLP encoder/decoder with a K-way softmax bottleneck.

Loss = recon + β_kl·mean_i KL(q_i||U) + β_ent·mean H(q_i) + β_batch·KL(q̄||U),
q̄ = mean_i q_i (배치 주변분포). β_ent↓ 뾰족, β_batch는 q̄→균등으로 클래스 붕괴 방지.
픽셀별 β_kl은 평평→ one-hot과 상충; 무감독 «분류» 모드에선 β_kl≈0 권장.

Decoder basis spectra: decode(one_hot_k) for k=0..K-1 (in the same input space as training, e.g. standardized).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class SpectrumSoftmaxAE(nn.Module):
    """input_dim -> 1024 -> 256 -> 128 -> K (logits) -> softmax -> decode -> input_dim."""

    def __init__(self, input_dim: int, latent_k: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_k = int(latent_k)
        k = self.latent_k
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, k),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.encoder(x)
        z = F.softmax(logits, dim=-1)
        recon = self.decoder(z)
        return recon, logits, z


def kl_softmax_uniform(logits: torch.Tensor) -> torch.Tensor:
    """Mean KL(q || U) over batch, q = softmax(logits), U = 1/K."""
    log_q = F.log_softmax(logits, dim=-1)
    q = log_q.exp()
    k = logits.shape[-1]
    return (q * (log_q + math.log(float(k)))).sum(dim=-1).mean()


def mean_softmax_entropy(z: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy (nats) of rows of z; z should be probabilities."""
    zc = z.clamp(min=1e-12)
    return (-(z * zc.log()).sum(dim=-1)).mean()


def kl_batch_marginal_uniform(z: torch.Tensor) -> torch.Tensor:
    """KL(q̄ || U) with q̄ = mean over batch of softmax rows, U discrete uniform on K classes."""
    bar = z.mean(dim=0).clamp(min=1e-12)
    k = z.shape[-1]
    log_bar = bar.log()
    return (bar * (log_bar + math.log(float(k)))).sum()


def train_spectrum_softmax_ae(
    y: np.ndarray,
    *,
    latent_k: int,
    random_state: int,
    standardize: bool,
    epochs: int,
    lr: float,
    beta_kl: float,
    beta_entropy: float,
    beta_batch_uniform: float,
    recon_loss: str,
    device: str | None,
) -> dict[str, Any]:
    """
    y: (n_wavelength, n_pixel) raw cube; trains on pixel rows.

    Returns numpy arrays and training diagnostics (no file I/O).
    """
    n_wl, n_pix = y.shape
    k = min(int(latent_k), n_pix, n_wl)
    if k < 1:
        raise ValueError("latent_k must be >= 1 and fit cube shape")

    X = y.T.astype(np.float64)
    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(random_state))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(random_state))

    model = SpectrumSoftmaxAE(n_wl, k).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    xt = torch.from_numpy(X_fit).to(device=dev, dtype=torch.float32)
    rl = (recon_loss or "mse").strip().lower()
    if rl == "mse":
        recon_fn = F.mse_loss
    elif rl in ("l1", "mae"):
        recon_fn = F.l1_loss
        rl = "l1"
    else:
        raise ValueError(f"recon_loss: {recon_loss!r} (use mse, l1)")

    last_recon = 0.0
    last_kl = 0.0
    last_ent = 0.0
    last_batch_kl = 0.0
    last_total = 0.0
    be = float(beta_entropy)
    bb = float(beta_batch_uniform)
    ep = int(epochs)
    for _ in range(ep):
        opt.zero_grad(set_to_none=True)
        recon, logits, z = model(xt)
        r = recon_fn(recon, xt)
        kl = kl_softmax_uniform(logits)
        ent = mean_softmax_entropy(z)
        bkl = kl_batch_marginal_uniform(z)
        loss = r + float(beta_kl) * kl + be * ent + bb * bkl
        loss.backward()
        opt.step()
        last_total = float(loss.detach().cpu())
        last_recon = float(r.detach().cpu())
        last_kl = float(kl.detach().cpu())
        last_ent = float(ent.detach().cpu())
        last_batch_kl = float(bkl.detach().cpu())

    model.eval()
    with torch.no_grad():
        recon_f, _, z_all = model(xt)
        z_np = z_all.cpu().numpy().astype(np.float64)
        recon_fit = recon_f.cpu().numpy().astype(np.float64)
        eye = torch.eye(k, device=dev, dtype=torch.float32)
        dec_space = model.decoder(eye).cpu().numpy().astype(np.float64)

    if scaler is not None:
        bases_raw = scaler.inverse_transform(dec_space)
        recon_raw = scaler.inverse_transform(recon_fit)
    else:
        bases_raw = dec_space
        recon_raw = recon_fit

    eps = 1e-12
    ent_per_pixel = -np.sum(z_np * np.log(np.clip(z_np, eps, 1.0)), axis=1)
    marginal = z_np.mean(axis=0)

    variances = np.var(z_np, axis=0)
    order = np.argsort(-variances)
    z_ord = z_np[:, order]
    bases_ord = bases_raw[order].T

    return {
        "latent_k": k,
        "z_ordered": z_ord,
        "bases_ordered": bases_ord,
        "variances_ordered": variances[order].astype(float).tolist(),
        "train_recon_last": float(last_recon),
        "recon_loss": rl,
        "train_kl_last": float(last_kl),
        "train_entropy_last": float(last_ent),
        "train_batch_marginal_kl_last": float(last_batch_kl),
        "train_loss_last": float(last_total),
        "standardize": bool(standardize),
        "device": str(dev),
        "epochs": ep,
        "lr": float(lr),
        "beta_kl": float(beta_kl),
        "beta_entropy": float(be),
        "beta_batch_uniform": float(bb),
        "marginal_latent_mean": marginal.astype(float).tolist(),
        "x_raw": X.astype(np.float64),
        "recon_raw": recon_raw.astype(np.float64),
        "latent_entropy_per_pixel": ent_per_pixel.astype(np.float64),
        "latent_argmax_per_pixel": z_np.argmax(axis=1).astype(np.int32),
        "latent_max_prob_per_pixel": z_np.max(axis=1).astype(np.float64),
    }
