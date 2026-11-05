"""Quick single run. Full reproduction: train_pinn.py, run_all.sh, analyze.py (see README)."""
import subprocess, sys
subprocess.check_call([sys.executable, 'train_pinn.py'])
subprocess.check_call([sys.executable, 'train.py', '--tag', 'proposed_s0', '--seed', '0'])
