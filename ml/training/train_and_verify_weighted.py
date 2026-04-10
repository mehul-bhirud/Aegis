import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import math

ROOT = Path(__file__).parent
TRAIN_TENSOR_PATH = ROOT / "train_tensor.pt"
TEST_TENSOR_PATH = ROOT / "test_tensor.pt"
MODEL_PATH = ROOT / "aegis_vae_model_weighted.pth"

INPUT_DIM = 61
LATENT_DIM = 10
EPOCHS = 75
BATCH_SIZE = 256
LR = 3e-4

class VariationalAutoencoder(nn.Module):
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

def weighted_mse_loss(recon_x, x):
    sq_error = (recon_x - x) ** 2
    threat_indices = [-1, -2, -3, -4, -5]
    weight_multiplier = 100.0
    for idx in threat_indices:
        sq_error[:, idx] *= weight_multiplier
    return sq_error.mean()

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = weighted_mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

@torch.no_grad()
def get_avg_weighted_mse(model, tensor):
    model.eval()
    recon, _, _ = model(tensor)
    return weighted_mse_loss(recon, tensor).item()

def print_ascii_ratio(ratio):
    print("\n" + "="*60)
    print("      FINAL SEPARATION RATIO")
    print("\n" + r"""
   _____ ______ _____          _____         _______ _____ ____  
  / ____|  ____|  __ \   /\   |  __ \     /\|__   __|_   _/ __ \ 
 | (___ | |__  | |__) | /  \  | |__) |   /  \  | |    | || |  | |
  \___ \|  __| |  ___/ / /\ \ |  _  /   / /\ \ | |    | || |  | |
  ____) | |____| |    / ____ \| | \ \  / ____ \| |   _| || |__| |
 |_____/|______|_|   /_/    \_\_|  \_\/_/    \_\_|  |_____\____/ 
                                                                 
    """ + "\n")
    print(f"                    >>> {ratio:5.2f}x <<<       ")
    print("="*60 + "\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Platform: {device}")

    # Load Data
    print("[*] Loading Tensors...")
    train_tensor = torch.load(TRAIN_TENSOR_PATH, map_location=device, weights_only=True)
    test_tensor = torch.load(TEST_TENSOR_PATH, map_location=device, weights_only=True)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    model = VariationalAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"[*] Training Model for {EPOCHS} Epochs...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}/{EPOCHS} -> Loss: {total_loss/len(train_loader):.4f}")

    print("\n[*] Training Complete. Verifying Data Separation...")
    train_mse = get_avg_weighted_mse(model, train_tensor)
    test_mse = get_avg_weighted_mse(model, test_tensor)

    print(f"    Baseline Mean Weighted MSE: {train_mse:.4f}")
    print(f"    Anomaly Mean Weighted MSE:  {test_mse:.4f}")

    ratio = test_mse / train_mse if train_mse > 0 else 0
    print_ascii_ratio(ratio)

    print(f"[*] Saving model to {MODEL_PATH.name}...")
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == '__main__':
    main()
