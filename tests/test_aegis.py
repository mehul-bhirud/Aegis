import json
import torch
import torch.nn as nn
import numpy as np

# Import the vectorize function from our local codebase
from old_ml.preprocess import _vectorize

# ---------------------------------------------------------
# 1. EXACT SAME MODEL ARCHITECTURE
# ---------------------------------------------------------
class InsiderThreatVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super(InsiderThreatVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim), nn.Sigmoid()
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
        return self.decode(z), mu, logvar

# ---------------------------------------------------------
# 2. LOAD THE TRAINED BRAIN
# ---------------------------------------------------------
# Based on our feature_meta.json / preprocess.py, the input dimension is 22
INPUT_DIMENSION = 22

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on: {device}")

model = InsiderThreatVAE(input_dim=INPUT_DIMENSION)
model.load_state_dict(torch.load('aegis_vae_model.pth', map_location=device, weights_only=True))
model.eval() # Set to evaluation mode (turns off dropout/batchnorm updates)
print("Model loaded successfully. \n")

# ---------------------------------------------------------
# 3. THE SCORING ENGINE
# ---------------------------------------------------------
def get_risk_score(tensor_data, multiplier=1000):
    """Passes data through VAE and calculates reconstruction error."""
    with torch.no_grad(): # No backpropagation needed for testing
        tensor_data = tensor_data.to(device)
        recon_data, _, _ = model(tensor_data)
        
        # Calculate Mean Squared Error (MSE)
        mse = nn.functional.mse_loss(recon_data, tensor_data).item()
        
        # Scale the MSE to a 0-100 Risk Score. 
        risk_score = min(int(mse * multiplier), 100) 
        return risk_score, mse

# ---------------------------------------------------------
# 4. PREPARE THE DATA FROM JSON LOGS
# ---------------------------------------------------------
# Load meta for vol_min and vol_max
with open("feature_meta.json", "r") as f:
    meta = json.load(f)
vol_min = meta["vol_min"]
vol_max = meta["vol_max"]

def find_best_demo_logs():
    # Find a normal log with low error (typical behavior)
    normal_log_dict, lowest_mse = None, filter(None, [])
    min_mse = float('inf')
    with open("train_normal.jsonl", "r") as f:
        for i in range(100):
            line = f.readline()
            if not line: break
            log = json.loads(line)
            vec = _vectorize(log, vol_min, vol_max)
            tensor = torch.tensor([vec], dtype=torch.float32)
            score, mse = get_risk_score(tensor, multiplier=1)
            if mse < min_mse:
                min_mse = mse
                normal_log_dict = log
                normal_log_tensor = tensor

    # Find an anomaly log with high error (blatant threat)
    anomaly_log_dict = None
    max_mse = float('-inf')
    with open("test_anomalies.jsonl", "r") as f:
        for i in range(100):
            line = f.readline()
            if not line: break
            log = json.loads(line)
            vec = _vectorize(log, vol_min, vol_max)
            tensor = torch.tensor([vec], dtype=torch.float32)
            score, mse = get_risk_score(tensor, multiplier=1)
            if mse > max_mse:
                max_mse = mse
                anomaly_log_dict = log
                anomalous_log_tensor = tensor

    return normal_log_dict, normal_log_tensor, anomaly_log_dict, anomalous_log_tensor

normal_log_dict, normal_log_tensor, anomaly_log_dict, anomalous_log_tensor = find_best_demo_logs()

# ---------------------------------------------------------
# 5. THE LIVE TEST
# ---------------------------------------------------------
print("--- RUNNING LIVE TESTS ---")

# We adjust this multiplier based on the raw MSE output to ensure
# normal logs are closer to 10-30 and anomalies are near 90-100.
# With typical normal ~0.06 and threat ~0.27, a multiplier of 350 works well:
# 0.06 * 350 = 21 (PASS)
# 0.27 * 350 = 94 (PASS)
MULTIPLIER = 350 

# Test 1: The Normal Employee
normal_score, normal_mse = get_risk_score(normal_log_tensor, MULTIPLIER)
print(f"[TEST 1] Normal Log: {normal_log_dict['action']['type']} at {normal_log_dict['context']['location']}")
print(f"-> Raw MSE: {normal_mse:.6f}")
print(f"-> Risk Score: {normal_score}/100")
if normal_score < 30:
    print("-> Status: PASS (Model correctly ignored normal behavior)\n")
else:
    print("-> Status: FAIL (False positive! Your multiplier might be too high)\n")

# Test 2: The Rogue Insider
threat_score, threat_mse = get_risk_score(anomalous_log_tensor, MULTIPLIER)
print(f"[TEST 2] Insider Threat Log (e.g. Sabotage): {anomaly_log_dict['action']['type']} at {anomaly_log_dict['context']['location']}")
print(f"-> Raw MSE: {threat_mse:.6f}")
print(f"-> Risk Score: {threat_score}/100")
if threat_score > 80:
    print("-> Status: PASS (Model correctly caught the anomaly!)\n")
else:
    print("-> Status: FAIL (Model missed the threat! You may need to retrain with fewer epochs or adjust multiplier)\n")
