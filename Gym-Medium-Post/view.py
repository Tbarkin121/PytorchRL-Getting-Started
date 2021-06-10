import os 

import torch as th
import pybullet_envs
import gym
import numpy as np

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

import base64
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder, sync_envs_normalization

from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.results_plotter import load_results, ts2xy

#Adding Stuff for new checkpoints
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps

#Import custom environment
import simple_driving


def main():
    # Create Storage Paths
    test_name = 'PPO-Linux'
    env_name = 'SimpleDriving-v0'
    videoName = 'videos'
    tb_log_name = test_name + '_' + env_name
    log_dir = os.path.join('log', test_name)
    num_cpu = 1
    model_stats_path = os.path.join(log_dir, 'Model_' + tb_log_name)
    env_stats_path = os.path.join(log_dir, 'Env_' + tb_log_name)
    checkpoint_path = os.path.join(log_dir, 'saved_models')
    best_path = os.path.join(log_dir, 'best_models')
    load_path = os.path.join(best_path, 'best_model.zip')
    video_path = os.path.join(log_dir, videoName)
    tb_log = os.path.join(log_dir, 'tb_log')
    
    total_timesteps = 100000

    # Create Environment
    # env = SubprocVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(num_cpu)])
    env = DummyVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(num_cpu)])
    env = VecNormalize.load(env_stats_path, env)
    env.reset()

    # Load Model
    model = PPO.load(load_path)
    model.set_env(env)

    
    obs = env.reset()
    for _ in range(total_timesteps):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)

    # Close the video recorder
    env.close()


def make_env(env_id: str, rank: int, seed: int = 1, log_dir=None) -> Callable:
    '''
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    '''
    def _init() -> gym.Env:
        env = gym.make(env_id)
        
        # Create folder if needed
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        
        env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

        

if __name__ == '__main__':
    main()