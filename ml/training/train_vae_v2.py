"""
train_vae_v2.py — Aegis-Fusion VAE Training & Evaluation Pipeline v2.0
======================================================================
Complete training and evaluation pipeline for the insider threat
detection Variational Autoencoder.

Architecture:
  Encoder:  input_dim → 128 → 64 → [μ, log σ²]  (latent_dim)
  Decoder:  latent_dim → 64 → 128 → input_dim    (Sigmoid)

Training strategy:
  • Trains ONLY on normal sessions  (train_tensor.pt)
  • β-annealing: KL weight ramps 0→1 over first 20% of epochs
  • Adam + ReduceLROnPlateau scheduler with early stopping

Risk scoring — Calibrated Sigmoid:
  ┌──────────────────────────────────────────────────────────────┐
  │  z = (MSE − μ_train) / σ_train          ← z-score          │
  │  risk = 100 / (1 + exp(−k·(z − c)))     ← calibrated sig.  │
  │                                                              │
  │  k, c are auto-tuned so that:                                │
  │    • train p95 (normal edge)   → risk ≈ 35                   │
  │    • test  p25 (weakest alert) → risk ≈ 85                   │
  └──────────────────────────────────────────────────────────────┘

Outputs:
  • aegis_vae_model_v2.pth   — model weights
  • threshold_stats.json     — calibration data for backend
  • anomaly_scores.json      — per-session risk breakdown

Usage:  python train_vae_v2.py
Deps:   torch, numpy  (pip install torch numpy)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Windows UTF-8 ─────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — all tuneable hyperparameters in one place
# ═══════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent

# Data paths
TRAIN_TENSOR_PATH = ROOT / "train_tensor.pt"
TEST_TENSOR_PATH  = ROOT / "test_tensor.pt"
TEST_META_PATH    = ROOT / "test_anomalies.jsonl"

# Output paths
MODEL_PATH    = ROOT / "aegis_vae_model_v2.pth"
THRESH_PATH   = ROOT / "threshold_stats.json"
SCORES_PATH   = ROOT / "anomaly_scores.json"

# Architecture  (auto-detected from tensor if possible)
INPUT_DIM  = 56          # overridden at runtime from tensor shape
LATENT_DIM = 10          # compressed representation size

# Training
EPOCHS       = 75
BATCH_SIZE   = 256
LR           = 3e-4
BETA_MAX     = 1.0       # maximum KL weight
ANNEAL_FRAC  = 0.20      # fraction of epochs for β warm-up
PATIENCE     = 12        # early stopping patience (epochs w/o improvement)

# Risk score auto-tuning targets
TARGET_LOW_RISK  = 35.0  # desired score for train p95
TARGET_HIGH_RISK = 85.0  # desired score for test  p25


# ═══════════════════════════════════════════════════════════════════════════
#  THE VAE ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class AegisVAE(nn.Module):
    """Variational Autoencoder for enterprise session anomaly detection.

    Architecture (default for 56-dim input):
        Encoder:  56 → 128 (LeakyReLU) → 64 (LeakyReLU) → μ(10), logσ²(10)
        Decoder:  10 → 64  (LeakyReLU) → 128 (LeakyReLU) → 56 (Sigmoid)

    LeakyReLU prevents dead neurons in the deeper architecture.
    Sigmoid output constrains reconstructions to [0, 1] matching
    the MinMax-normalised inputs from preprocess.py.
    """

    def __init__(self, input_dim: int = INPUT_DIM,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # ── Decoder ───────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),          # output ∈ [0, 1]
        )

        # Xavier init for smoother early training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(μ, σ²) using the reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ═══════════════════════════════════════════════════════════════════════════
#  VAE LOSS FUNCTION  (Reconstruction + KL Divergence)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_weighted_mse(recon_x: torch.Tensor, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    sq_error = (recon_x - x) ** 2
    threat_indices = [-1, -2, -3, -4, -5] 
    weight_multiplier = 100.0 
    
    for idx in threat_indices:
        sq_error[:, idx] *= weight_multiplier
        
    if reduction == 'none':
        return sq_error.mean(dim=1)
    return sq_error.mean()

def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, float, float]:
    """
    Combined VAE loss with controllable β weight.

    Loss = MSE(recon, x) + β · KL(q(z|x) ‖ p(z))

    MSE  = Mean Squared Error — penalises bad reconstructions.
           If the model can't rebuild the input, it hasn't learned it.

    KL   = -0.5 · Σ(1 + log σ² − μ² − σ²)
           Pushes the latent distribution toward N(0, 1).
           This regulariser ensures similar inputs cluster together,
           making anomalies stand out as outliers.

    β    = KL weight.  During annealing (β < 1), the model focuses on
           reconstruction quality first.  At β = 1, we get the standard
           ELBO objective.

    Returns: (total_loss, recon_loss_value, kl_loss_value)
    """
    recon_loss = calculate_weighted_mse(recon, x, reduction="mean")

    # Closed-form KL divergence for diagonal Gaussian q(z|x) vs N(0,1)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()


# ═══════════════════════════════════════════════════════════════════════════
#  RISK SCORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_sigmoid(
    train_mses: np.ndarray,
    test_mses: np.ndarray,
) -> tuple[float, float, float, float]:
    """Auto-tune the sigmoid risk function from data distributions.

    We solve for (steepness, center) such that:
      • train p95  →  risk ≈ TARGET_LOW_RISK   (~35)
      • test  p25  →  risk ≈ TARGET_HIGH_RISK  (~85)

    Math:
      risk(z) = 100 / (1 + exp(-k·(z − c)))
      To invert:  k·(z − c) = logit(risk/100)
                              = ln(risk / (100 − risk))

      Two equations, two unknowns:
        k·(z_low − c)  = logit(TARGET_LOW  / 100)    … (1)
        k·(z_high − c) = logit(TARGET_HIGH / 100)    … (2)

      Subtract (1) from (2):
        k·(z_high − z_low) = logit_high − logit_low
        k = (logit_high − logit_low) / (z_high − z_low)

      Back-substitute for c:
        c = z_low − logit_low / k

    Returns: (mean, std, steepness, center)
    """
    mean = float(np.mean(train_mses))
    std  = float(np.std(train_mses))
    if std < 1e-12:
        std = 1e-6

    z_train = (train_mses - mean) / std
    z_test  = (test_mses  - mean) / std

    z_low  = float(np.percentile(z_train, 95))     # normal edge
    z_high = float(np.percentile(z_test,  25))      # weakest anomaly

    logit_low  = math.log(TARGET_LOW_RISK  / (100 - TARGET_LOW_RISK))
    logit_high = math.log(TARGET_HIGH_RISK / (100 - TARGET_HIGH_RISK))

    dz = z_high - z_low
    if dz > 0.05:
        steepness = (logit_high - logit_low) / dz
        center    = z_low - logit_low / steepness
    else:
        # Poor separation fallback — use conservative defaults
        steepness = 1.5
        center    = 2.0

    return mean, std, steepness, center


def compute_risk_score(mse: float, mean: float, std: float,
                       steepness: float, center: float) -> int:
    """Map raw MSE → 0-100 integer risk score via calibrated sigmoid.

    The sigmoid is centred at `center` standard deviations above the
    training mean.  `steepness` controls transition sharpness.

    Diagram:
      100 ┤                              ╭──────
          │                           ╭──╯
          │                        ╭──╯
       50 ┤ · · · · · · · · · ·╭──╯
          │                 ╭──╯
          │              ╭──╯
        0 ┤──────────────╯
          └──┬────┬────┬────┬────┬────┬──→ z-score
            -1    0    1    2    3    4
                        ↑center
    """
    z = (mse - mean) / std if std > 0 else 0.0
    score = 100.0 / (1.0 + math.exp(-steepness * (z - center)))
    return max(0, min(100, int(round(score))))


# ═══════════════════════════════════════════════════════════════════════════
#  PER-SESSION MSE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_mse_distribution(model: AegisVAE, tensor: torch.Tensor,
                             device: torch.device) -> np.ndarray:
    """Compute per-session reconstruction error (MSE) for an entire tensor."""
    model.eval()
    mses = []
    # Process in batches to avoid OOM on large tensors
    loader = DataLoader(TensorDataset(tensor), batch_size=1024, shuffle=False)
    for (batch,) in loader:
        batch = batch.to(device)
        recon, _, _ = model(batch)
        # Per-sample weighted MSE (not averaged across batch)
        sample_mse = calculate_weighted_mse(recon, batch, reduction='none')
        mses.append(sample_mse.cpu().numpy())
    return np.concatenate(mses)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.perf_counter()

    print("""
╔══════════════════════════════════════════════════════════════╗
║  AEGIS-FUSION  VAE Training Pipeline v2.0                   ║
║  Insider Threat Detection — Session-Level Anomaly Scoring   ║
╚══════════════════════════════════════════════════════════════╝""")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ──────────────────────────────────────────────────────────────────
    # STEP 1 — Load Tensors
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading tensors …")

    train_data = torch.load(TRAIN_TENSOR_PATH, map_location="cpu",
                            weights_only=True)
    test_data  = torch.load(TEST_TENSOR_PATH,  map_location="cpu",
                            weights_only=True)

    input_dim = train_data.shape[1]
    print(f"      train_tensor : {list(train_data.shape)}  "
          f"({train_data.shape[0]:,} normal sessions)")
    print(f"      test_tensor  : {list(test_data.shape)}  "
          f"({test_data.shape[0]:,} anomaly sessions)")
    print(f"      input_dim    : {input_dim}  (auto-detected)")
    print(f"      device       : {device}")

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    # ──────────────────────────────────────────────────────────────────
    # STEP 2 — Build Model
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[2/5] Building VAE  (input={input_dim} → latent={LATENT_DIM}) …")

    model = AegisVAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters   : {params:,}")
    print(f"      Architecture : {input_dim}→128→64→[μ,logσ²]→{LATENT_DIM}"
          f"→64→128→{input_dim}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ──────────────────────────────────────────────────────────────────
    # STEP 3 — Training Loop (β-annealed VAE)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[3/5] Training for {EPOCHS} epochs  "
          f"(β-anneal over first {int(ANNEAL_FRAC * 100)}%) …")
    print(f"      {'Epoch':>7s} │ {'Loss':>10s} │ {'Recon':>10s} │ "
          f"{'KL':>10s} │ {'β':>6s} │ {'LR':>10s}")
    print(f"      {'─' * 7} │ {'─' * 10} │ {'─' * 10} │ "
          f"{'─' * 10} │ {'─' * 6} │ {'─' * 10}")

    anneal_epochs = max(1, int(EPOCHS * ANNEAL_FRAC))
    best_loss  = float("inf")
    patience_c = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # β annealing: linearly ramp from 0 → BETA_MAX
        beta = min(BETA_MAX, BETA_MAX * epoch / anneal_epochs)

        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl    = 0.0
        n_batches   = 0

        for (batch,) in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)

            loss, rl, kl = vae_loss(recon, batch, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += rl
            epoch_kl    += kl
            n_batches   += 1

        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl    = epoch_kl    / n_batches
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(avg_total)

        # Early stopping check
        if avg_total < best_loss - 1e-6:
            best_loss  = avg_total
            patience_c = 0
            # Save best model state
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_c += 1

        # Print every 5 epochs (and first/last)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"      {epoch:>4d}/{EPOCHS:<2d} │ {avg_total:>10.6f} │ "
                  f"{avg_recon:>10.6f} │ {avg_kl:>10.6f} │ "
                  f"{beta:>6.3f} │ {current_lr:>10.2e}")

        if patience_c >= PATIENCE:
            print(f"\n      ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()
    print(f"      Best loss    : {best_loss:.6f}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4 — Evaluation & Risk Score Calibration
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[4/5] Evaluating & calibrating risk scores …")

    # Compute MSE distributions
    train_mses = compute_mse_distribution(model, train_data, device)
    test_mses  = compute_mse_distribution(model, test_data,  device)

    # Statistics
    t_mean, t_std  = float(np.mean(train_mses)), float(np.std(train_mses))
    t_p50, t_p95, t_p99 = [float(np.percentile(train_mses, p))
                            for p in (50, 95, 99)]
    a_mean = float(np.mean(test_mses))
    a_min, a_max = float(np.min(test_mses)), float(np.max(test_mses))

    separation = a_mean / t_mean if t_mean > 0 else 0
    print(f"      ┌─────────────────────────────────────────────────┐")
    print(f"      │  TRAIN MSE (Normal baseline)                    │")
    print(f"      │    mean   = {t_mean:.6f}                        │")
    print(f"      │    std    = {t_std:.6f}                         │")
    print(f"      │    p50    = {t_p50:.6f}                         │")
    print(f"      │    p95    = {t_p95:.6f}                         │")
    print(f"      │    p99    = {t_p99:.6f}                         │")
    print(f"      ├─────────────────────────────────────────────────┤")
    print(f"      │  TEST MSE  (Kill-shot sessions)                 │")
    print(f"      │    mean   = {a_mean:.6f}                        │")
    print(f"      │    min    = {a_min:.6f}                         │")
    print(f"      │    max    = {a_max:.6f}                         │")
    print(f"      ├─────────────────────────────────────────────────┤")
    print(f"      │  Separation ratio : {separation:.2f}x             │")
    print(f"      └─────────────────────────────────────────────────┘")

    # Auto-tune sigmoid
    mean, std, steepness, center = calibrate_sigmoid(train_mses, test_mses)
    print(f"\n      Auto-tuned sigmoid parameters:")
    print(f"        steepness (k)  = {steepness:.4f}")
    print(f"        center    (c)  = {center:.4f}  "
          f"({center:.1f}σ above training mean)")

    # Score all sessions
    train_scores = np.array([
        compute_risk_score(m, mean, std, steepness, center)
        for m in train_mses
    ])
    test_scores = np.array([
        compute_risk_score(m, mean, std, steepness, center)
        for m in test_mses
    ])

    print(f"\n      Risk score distribution:")
    print(f"        Train (normal):  μ={np.mean(train_scores):5.1f}  "
          f"median={np.median(train_scores):5.1f}  "
          f"p95={np.percentile(train_scores, 95):5.1f}  "
          f"max={np.max(train_scores)}")
    print(f"        Test  (anomaly): μ={np.mean(test_scores):5.1f}  "
          f"median={np.median(test_scores):5.1f}  "
          f"p25={np.percentile(test_scores, 25):5.1f}  "
          f"min={np.min(test_scores)}")

    # Detection rates at key thresholds
    for threshold in [50, 70, 85]:
        n_flagged = int(np.sum(test_scores >= threshold))
        rate = n_flagged / len(test_scores) * 100
        fp   = int(np.sum(train_scores >= threshold))
        fp_r = fp / len(train_scores) * 100
        print(f"        @threshold={threshold:>3d}:  "
              f"detect={rate:5.1f}% ({n_flagged:,}/{len(test_scores):,})  "
              f"FP={fp_r:.2f}% ({fp:,}/{len(train_scores):,})")

    # ── Per-scenario breakdown ────────────────────────────────────────
    if TEST_META_PATH.exists():
        print(f"\n      Per-scenario breakdown:")
        test_meta = []
        with open(TEST_META_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                test_meta.append(json.loads(line))

        scenario_scores: dict[str, list[int]] = {}
        for i, meta in enumerate(test_meta):
            score = int(test_scores[i])
            for ks_type in meta.get("killshot_types", []):
                scenario_scores.setdefault(ks_type, []).append(score)

        print(f"        {'Scenario':<20s} │ {'Count':>6s} │ "
              f"{'Mean':>6s} │ {'Min':>5s} │ {'Max':>5s} │ {'≥85':>5s}")
        print(f"        {'─' * 20} │ {'─' * 6} │ "
              f"{'─' * 6} │ {'─' * 5} │ {'─' * 5} │ {'─' * 5}")
        for name in ["analog_hole", "steganography", "retail_fraud",
                      "honey_token", "shadow_admin"]:
            scores = scenario_scores.get(name, [])
            if scores:
                arr = np.array(scores)
                crit = int(np.sum(arr >= 85))
                print(f"        {name:<20s} │ {len(arr):>6,} │ "
                      f"{np.mean(arr):>6.1f} │ {np.min(arr):>5d} │ "
                      f"{np.max(arr):>5d} │ {crit:>4d}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 5 — Export Artefacts
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving artefacts …")

    # Model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"      → {MODEL_PATH.name}  ({params:,} params)")

    # Threshold / calibration stats for the backend
    threshold_data = {
        "train_mse_mean":  t_mean,
        "train_mse_std":   t_std,
        "train_mse_p50":   t_p50,
        "train_mse_p95":   t_p95,
        "train_mse_p99":   t_p99,
        "test_mse_mean":   a_mean,
        "sigmoid_steepness": steepness,
        "sigmoid_center":    center,
        "separation_ratio":  round(separation, 4),
        "input_dim":         input_dim,
        "latent_dim":        LATENT_DIM,
        "epochs_trained":    epoch,
        "best_loss":         best_loss,
    }
    THRESH_PATH.write_text(
        json.dumps(threshold_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"      → {THRESH_PATH.name}")

    # Per-session anomaly scores (for frontend dashboard)
    anomaly_output: list[dict] = []
    if TEST_META_PATH.exists() and test_meta:
        for i, meta in enumerate(test_meta):
            anomaly_output.append({
                "session_id":    meta["session_id"],
                "risk_score":    int(test_scores[i]),
                "raw_mse":       round(float(test_mses[i]), 8),
                "killshot_types": meta.get("killshot_types", []),
                "log_count":     meta.get("log_count", 0),
            })
    else:
        for i in range(len(test_mses)):
            anomaly_output.append({
                "session_id":  f"test_{i}",
                "risk_score":  int(test_scores[i]),
                "raw_mse":     round(float(test_mses[i]), 8),
            })

    SCORES_PATH.write_text(
        json.dumps(anomaly_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"      → {SCORES_PATH.name}  ({len(anomaly_output):,} sessions)")

    # ── Final Summary ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    crit_detected = int(np.sum(test_scores >= 85))
    crit_rate     = crit_detected / len(test_scores) * 100

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  TRAINING COMPLETE                                          ║
╠══════════════════════════════════════════════════════════════╣
║  Model           : AegisVAE ({input_dim}→{LATENT_DIM} latent){"":>20s}║
║  Parameters      : {params:>10,}                              ║
║  Epochs          : {epoch:>10} / {EPOCHS:<10}                   ║
║  Best loss       : {best_loss:>10.6f}                              ║
║  Separation      : {separation:>10.2f}x                             ║
║  Critical detect : {crit_rate:>9.1f}%  ({crit_detected:,}/{len(test_scores):,}){"":>14s}║
║  Time            : {elapsed:>10.1f}s                               ║
╚══════════════════════════════════════════════════════════════╝
""")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
