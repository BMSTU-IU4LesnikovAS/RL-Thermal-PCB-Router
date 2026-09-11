import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

from pcb_env import PCBRouterEnv
from pinn_thermal import ThermalPINN, pretrain_pinn, device


if __name__ == "__main__":
    print("Pretraining PINN surrogate on the heat-conduction equation...", flush=True)
    pinn = ThermalPINN()
    pinn = pretrain_pinn(pinn, q_base=50.0, k_copper=385.0, T_boundary=290.0, epochs=500)

    env = PCBRouterEnv(grid_size=32, pinn_model=pinn)
    check_env(env)
    print("Environment check passed.", flush=True)

    model = PPO("MlpPolicy", env,
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=256,
                gamma=0.99,
                clip_range=0.2,
                verbose=1,
                device="cpu")

    print("Starting PPO training...", flush=True)
    model.learn(total_timesteps=50000)

    print("Training complete. Building the optimal trace...", flush=True)
    obs, _ = env.reset()
    done = False
    steps = 0
    action_log = []
    while not done and steps < 500:
        action, _states = model.predict(obs, deterministic=True)
        action_log.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    print(f"Rollout finished after {steps} steps.", flush=True)
    print(f"Final position: {env.current_pos}, target: {env.target_pos}", flush=True)
    print(f"Path length: {len(env.path)}", flush=True)
    print(f"Actions taken (first 30): {action_log[:30]}", flush=True)
    print(f"Final reward: {reward}", flush=True)

    plt.figure(figsize=(6, 6))
    plt.title("RL-Optimized PCB Layout")
    plt.imshow(obs[0] + obs[1] * 0.5, cmap='hot')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.savefig('optimized_layout.png', dpi=150)
    print("Saved plot to optimized_layout.png — script finished.", flush=True)
