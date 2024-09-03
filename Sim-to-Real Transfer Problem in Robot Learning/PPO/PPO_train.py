from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-env', type=str, choices=['source', 'target'], required=True, help='Environment to train in')
    parser.add_argument('--eval-env', type=str, choices=['source', 'target'], required=True, help='Environment to evaluate in')
    parser.add_argument('--source-log-path', type=str, required=True, help='Path to log data for the source environment')
    parser.add_argument('--target-log-path', type=str, required=True, help='Path to log data for the target environment')
    parser.add_argument('--timesteps', type=int, required=True, help='Number of timesteps for training')
    return parser.parse_args()

args = parse_args()

N_ENVS = os.cpu_count()

def main():
    source_env = make_vec_env('CustomHopper-source-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    target_env = make_vec_env('CustomHopper-target-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

    if args.train_env == 'source':
        train_env = source_env
        eval_log_path = args.source_log_path
    else:
        train_env = target_env
        eval_log_path = args.target_log_path

    if args.eval_env == 'source':
        eval_env = source_env
    else:
        eval_env = target_env

    checkpoint_callback = CheckpointCallback(save_freq=int(np.ceil(1e7 / 12)), save_path='./', name_prefix="model_"+args.train_env+"_")
    eval_callback = EvalCallback(eval_env=eval_env, n_eval_episodes=50, eval_freq=15000, log_path=eval_log_path)  # Evaluation during training
    callback_list = [checkpoint_callback, eval_callback]

    model = PPO('MlpPolicy', env=train_env, n_steps=1024, batch_size=128, learning_rate=0.00025, verbose=1, device='cpu', tensorboard_log="./ppo_train_tensorboard/")

    callback = CallbackList(callback_list)
    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name=args.train_env)
    model.save(f"ppo_model_{args.train_env}_{args.timesteps}")

    # Evaluate in the specified evaluation environment after training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)

    print(f"Mean reward in {args.eval_env} environment: {mean_reward} +/- {std_reward}")

if __name__ == '__main__':
    main()