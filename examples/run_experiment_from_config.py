"""Run a full experiment from a config file: generalized-Newtonian obstacle channel."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "work" / "examples" / "run_experiment_from_config"
out.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, str(root / "experiments" / "gnf_truth.py"),
    "--config", str(root / "experiments" / "configs" / "gnf_obstacle.yaml"),
    "--output-dir", str(out),
]
print("run_experiment_from_config", " ".join(cmd))
rc = subprocess.call(cmd, cwd=str(root))
print("run_experiment_from_config rc", rc)
sys.exit(rc)
