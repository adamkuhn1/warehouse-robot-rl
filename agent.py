import gymnasium as gym
from airplane_boarding import WarehouseEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

model_dir = "models"
log_dir = "logs"

def train():
    # Create vectorized environments for parallel training
    env = make_vec_env(WarehouseEnv, n_envs=4, env_kwargs={"grid_size":6, "n_robots":2, "n_items":2}, vec_env_cls=SubprocVecEnv)
    model = PPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    eval_callback = EvalCallback(
        env,
        eval_freq=10_000,
        verbose=1,
        best_model_save_path=os.path.join(model_dir, 'PPO_Warehouse'),
    )
    model.learn(total_timesteps=int(1e6), callback=eval_callback)

def test_sb3(model_timesteps, renders=True):
    # Test a trained model and render the environment
    env = WarehouseEnv(render_mode="terminal" if renders else None)
    model = PPO.load(f"models/ppo_warehouse_{model_timesteps}", env=env)
    rewards = 0
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        rewards += reward
        env.render()
    print(f"Total rewards: {rewards}")

if __name__ == "__main__":
    train()
    # To test a trained model, uncomment the line below and specify the checkpoint
    # test_sb3(100_000)