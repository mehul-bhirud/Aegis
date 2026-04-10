import torch
import torch.nn as nn
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "aegis_vae_model_v2.pth"
TRAIN_TENSOR_PATH = ROOT / "train_tensor.pt"
TEST_TENSOR_PATH = ROOT / "test_tensor.pt"
THRESH_PATH = ROOT / "threshold_stats.json"

INPUT_DIM = 61
LATENT_DIM = 10

class AegisVAE(nn.Module):
    """Variational Autoencoder used to determine reconstruction error (MSE)."""
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def calculate_weighted_mse(recon_x: torch.Tensor, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    sq_error = (recon_x - x) ** 2
    threat_indices = [-1, -2, -3, -4, -5] 
    weight_multiplier = 100.0 
    
    for idx in threat_indices:
        sq_error[:, idx] *= weight_multiplier
        
    if reduction == 'none':
        return sq_error.mean(dim=1)
    return sq_error.mean()

@torch.no_grad()
def get_mse(model, tensor):
    """Passes tensor through model and returns raw Mean Squared Error per item."""
    model.eval()
    recon, _, _ = model(tensor)
    # Weighted Mean Squared Error: average across the feature dimension
    sample_mse = calculate_weighted_mse(recon, tensor, reduction='none')
    return sample_mse

def mse_to_risk_score(mse: float, mean: float, std: float, steepness: float, center: float) -> int:
    """Scales the raw MSE mathematically into a human-readable 0-100 Risk Score."""
    z = (mse - mean) / std if std > 0 else 0.0
    score = 100.0 / (1.0 + math.exp(-steepness * (z - center)))
    return max(0, min(100, int(round(score))))

def main():
    print("=========================================================")
    print("  AEGIS-FUSION: MATHEMATICAL PERFORMANCE VERIFICATION  ")
    print("=========================================================\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Platform: {device}")

    # 1. Load the Model
    print(f"[*] Loading model from {MODEL_PATH.name}...")
    model = AegisVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 2. Load Train Baseline (Sample 5000 rows)
    print(f"\n[*] Loading sample of 5,000 baseline items from {TRAIN_TENSOR_PATH.name}...")
    train_tensor = torch.load(TRAIN_TENSOR_PATH, map_location='cpu', weights_only=True)[:5000].to(device)
    
    # 3. Load Test Anomalies
    print(f"[*] Loading test items from {TEST_TENSOR_PATH.name}...")
    test_tensor = torch.load(TEST_TENSOR_PATH, map_location='cpu', weights_only=True).to(device)

    # Calculate MSEs
    train_mses = get_mse(model, train_tensor)
    test_mses = get_mse(model, test_tensor)

    train_mean = train_mses.mean().item()
    test_mean = test_mses.mean().item()

    print(f"\n-> Baseline Mean MSE (Normal):    {train_mean:.6f}")
    print(f"-> Target Test MSE (Anomalies):   {test_mean:.6f}")

    # 4. Separation Ratio
    separation = test_mean / train_mean if train_mean > 0 else 0
    print(f"=========================================================")
    print(f"   SEPARATION RATIO: {separation:.2f}x")
    print(f"=========================================================\n")

    # 5. Load Params & Scale Risk Scores
    print("[*] Applying dataset-tuned Sigmoid scaling...")
    stats = json.loads(THRESH_PATH.read_text())
    s_mean, s_std = stats["train_mse_mean"], stats["train_mse_std"]
    k, c = stats["sigmoid_steepness"], stats["sigmoid_center"]

    safe = 0
    warning = 0
    critical = 0

    for mse in test_mses.tolist():
        score = mse_to_risk_score(mse, s_mean, s_std, k, c)
        if score <= 30:
            safe += 1
        elif score <= 75:
            warning += 1
        else:
            critical += 1

    print("\n--- TEST DATASET RISK SCORE DISTRIBUTION ---")
    print(f"Safe (0-30)            : {safe:5d} sessions")
    print(f"Noise/Warning (31-75)  : {warning:5d} sessions")
    print(f"Critical/Kill (76-100) : {critical:5d} sessions")
    print(f"Total Test Cases       : {len(test_mses):5d}")

if __name__ == '__main__':
    main()
