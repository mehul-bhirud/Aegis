"""
Steps 2-4 – VAE Training & Inference Pipeline
===============================================
  Step 2: InsiderThreatVAE architecture + loss function
  Step 3: Training loop on train_tensor.pt (normal logs only)
  Step 4: Inference – score every anomaly in test_tensor.pt

Inputs  (from preprocess.py):
  train_tensor.pt     – [499500 × 22] normal activity
  test_tensor.pt      – [500 × 22]    kill-shot anomalies
  feature_meta.json   – column names & scaling params

Outputs:
  aegis_vae_model.pth           – trained model weights
  anomaly_scores.json           – per-log risk scores for test set
  threshold_stats.json          – calibration stats for production
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent
TRAIN_T     = ROOT / "train_tensor.pt"
TEST_T      = ROOT / "test_tensor.pt"
META_F      = ROOT / "feature_meta.json"
MODEL_PATH  = ROOT / "aegis_vae_model.pth"
SCORES_PATH = ROOT / "anomaly_scores.json"
THRESH_PATH = ROOT / "threshold_stats.json"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LATENT_DIM  = 5
BATCH_SIZE  = 256        # larger batch → faster on CPU with 500k rows
EPOCHS      = 20
LR          = 1e-3
RISK_MULTIPLIER = 1000   # tuning knob for 0-100 risk mapping


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 – THE VAE ARCHITECTURE                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class InsiderThreatVAE(nn.Module):
    """Variational Autoencoder for enterprise activity anomaly detection.

    Encoder compresses the F-dimensional feature vector down to a
    `latent_dim`-dimensional latent space.  Decoder reconstructs.
    Anomalies produce high reconstruction error because the model has
    never seen patterns like them during training.
    """

    def __init__(self, input_dim: int, latent_dim: int = 5):
        super().__init__()

        # Encoder: input_dim → 32 → 16 → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

        # Decoder: latent_dim → 16 → 32 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),           # squashes output to [0, 1]
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --- Loss function ---
def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reconstruction (MSE) + KL-Divergence."""
    mse = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 – TRAINING LOOP                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def train(model: InsiderThreatVAE, dataloader: DataLoader,
          device: torch.device, epochs: int = EPOCHS) -> list[float]:
    """Train the VAE on normal-only data.  Returns per-epoch avg losses."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    history: list[float] = []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader.dataset)
        history.append(avg)
        print(f"  Epoch {epoch:2d}/{epochs}  │  Loss: {avg:.6f}")

    return history


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 – INFERENCE & RISK SCORING                                    ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def compute_reconstruction_errors(model: InsiderThreatVAE,
                                   tensor: torch.Tensor,
                                   device: torch.device) -> np.ndarray:
    """Return per-sample MSE reconstruction error (numpy array)."""
    model.eval()
    with torch.no_grad():
        t = tensor.to(device)
        recon, _, _ = model(t)
        # per-sample MSE (mean over features, not sum)
        errors = ((recon - t) ** 2).mean(dim=1).cpu().numpy()
    return errors


def errors_to_risk(errors: np.ndarray, multiplier: float = RISK_MULTIPLIER) -> list[int]:
    """Map raw MSE errors to 0-100 integer risk scores."""
    return [min(int(e * multiplier), 100) for e in errors]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load tensors -----------------------------------------------------
    print("[*] Loading tensors …")
    X_train = torch.load(TRAIN_T, weights_only=True)
    X_test  = torch.load(TEST_T,  weights_only=True)
    meta    = json.loads(META_F.read_text())
    input_dim = X_train.shape[1]
    print(f"    train : {list(X_train.shape)}")
    print(f"    test  : {list(X_test.shape)}")
    print(f"    features: {input_dim}")

    # ---- Build model ------------------------------------------------------
    model = InsiderThreatVAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    model params: {total_params:,}")

    # ---- DataLoader -------------------------------------------------------
    dataset    = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=(device.type == "cuda"))

    # ---- Train ------------------------------------------------------------
    print(f"\n{'─'*50}")
    print(f"  TRAINING  │  {EPOCHS} epochs  │  batch {BATCH_SIZE}")
    print(f"{'─'*50}")
    t0 = time.perf_counter()
    history = train(model, dataloader, device, EPOCHS)
    elapsed = time.perf_counter() - t0
    print(f"{'─'*50}")
    print(f"  Training complete in {elapsed:.1f}s")

    # ---- Save model -------------------------------------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH.name}")

    # ---- Inference on TRAIN (calibration) ---------------------------------
    print(f"\n[*] Calibrating threshold on training data …")
    train_errors = compute_reconstruction_errors(model, X_train, device)
    train_mean   = float(np.mean(train_errors))
    train_std    = float(np.std(train_errors))
    train_p99    = float(np.percentile(train_errors, 99))
    train_max    = float(np.max(train_errors))
    print(f"    Train MSE  mean={train_mean:.6f}  std={train_std:.6f}")
    print(f"    Train MSE  p99={train_p99:.6f}   max={train_max:.6f}")

    # ---- Inference on TEST (anomalies) ------------------------------------
    print(f"\n[*] Scoring {X_test.shape[0]} anomaly logs …")
    test_errors = compute_reconstruction_errors(model, X_test, device)
    test_risks  = errors_to_risk(test_errors)

    test_mean = float(np.mean(test_errors))
    test_std  = float(np.std(test_errors))
    test_min  = float(np.min(test_errors))
    test_max  = float(np.max(test_errors))

    print(f"    Anomaly MSE  mean={test_mean:.6f}  std={test_std:.6f}")
    print(f"    Anomaly MSE  min={test_min:.6f}   max={test_max:.6f}")

    risk_arr = np.array(test_risks)
    print(f"    Risk scores  mean={risk_arr.mean():.1f}  min={risk_arr.min()}  max={risk_arr.max()}")
    above_85 = int(np.sum(risk_arr > 85))
    above_50 = int(np.sum(risk_arr > 50))
    print(f"    Alerts (risk > 85): {above_85}/{len(test_risks)}")
    print(f"    Warns  (risk > 50): {above_50}/{len(test_risks)}")

    # ---- Save scores ------------------------------------------------------
    # Reload test JSONL to attach session_id for traceability
    test_jsonl = ROOT / "test_anomalies.jsonl"
    test_logs = []
    if test_jsonl.exists():
        with open(test_jsonl, encoding="utf-8") as fh:
            test_logs = [json.loads(line) for line in fh]

    scored_output = []
    for idx in range(len(test_risks)):
        entry = {
            "index": idx,
            "risk_score": test_risks[idx],
            "reconstruction_error": round(float(test_errors[idx]), 8),
        }
        if idx < len(test_logs):
            entry["session_id"] = test_logs[idx].get("session_id", "")
            entry["event_id"]   = test_logs[idx].get("event_id", "")
            entry["timestamp"]  = test_logs[idx].get("timestamp", "")
            entry["actor"]      = test_logs[idx].get("actor", {})
            entry["action"]     = test_logs[idx].get("action", {})
            entry["resource"]   = test_logs[idx].get("resource", {})
        scored_output.append(entry)

    # Sort by risk descending
    scored_output.sort(key=lambda x: x["risk_score"], reverse=True)

    with open(SCORES_PATH, "w", encoding="utf-8") as fh:
        json.dump(scored_output, fh, indent=2, ensure_ascii=False)
    print(f"\n  Scores saved → {SCORES_PATH.name}")

    # ---- Save threshold stats --------------------------------------------
    thresh = {
        "train_mse_mean": train_mean,
        "train_mse_std": train_std,
        "train_mse_p99": train_p99,
        "train_mse_max": train_max,
        "anomaly_mse_mean": test_mean,
        "anomaly_mse_std": test_std,
        "anomaly_mse_min": test_min,
        "anomaly_mse_max": test_max,
        "risk_multiplier": RISK_MULTIPLIER,
        "recommended_threshold": 85,
        "alerts_above_85": above_85,
        "warns_above_50": above_50,
        "training_history": history,
        "epochs": EPOCHS,
        "latent_dim": LATENT_DIM,
        "input_dim": input_dim,
        "device": str(device),
        "training_time_s": round(elapsed, 1),
    }
    with open(THRESH_PATH, "w", encoding="utf-8") as fh:
        json.dump(thresh, fh, indent=2)
    print(f"  Threshold stats → {THRESH_PATH.name}")

    # ---- Top 10 riskiest logs --------------------------------------------
    print(f"\n{'='*60}")
    print(f"  TOP 10 RISKIEST ANOMALIES")
    print(f"{'='*60}")
    for i, s in enumerate(scored_output[:10]):
        sid = s.get("session_id", "?")
        ts  = s.get("timestamp", "?")
        act = s.get("action", {}).get("type", "?")
        res = s.get("resource", {}).get("name", "?")
        print(f"  #{i+1:2d}  risk={s['risk_score']:3d}  "
              f"err={s['reconstruction_error']:.6f}  "
              f"{sid:15s}  {act:14s}  {res}")

    # ---- Separation ratio -------------------------------------------------
    sep = test_mean / train_mean if train_mean > 0 else float("inf")
    print(f"\n  Separation ratio (anomaly/normal MSE): {sep:.1f}x")
    if sep > 5:
        print("  ✓ Excellent separation – model clearly distinguishes anomalies")
    elif sep > 2:
        print("  ~ Decent separation – consider tuning latent_dim or epochs")
    else:
        print("  ✗ Poor separation – review feature engineering")


if __name__ == "__main__":
    main()
