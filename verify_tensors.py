"""Quick verification of preprocessing outputs."""
import os, torch

ROOT = os.path.dirname(__file__)
files = ["train_tensor.pt", "test_tensor.pt", "test_anomalies.jsonl", "feature_meta.json"]

print("=" * 60)
print("  TENSOR VERIFICATION")
print("=" * 60)

for f in files:
    path = os.path.join(ROOT, f)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        if f.endswith(".pt"):
            t = torch.load(path, weights_only=True)
            print(f"  [OK] {f:30s}  shape={str(list(t.shape)):>16s}  {size_kb:>8.1f} KB")
        else:
            print(f"  [OK] {f:30s}  {'---':>16s}  {size_kb:>8.1f} KB")
    else:
        print(f"  [!!] {f:30s}  MISSING")

print("=" * 60)
