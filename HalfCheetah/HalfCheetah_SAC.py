import os 

import pybullet_envs
import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

import base64
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.results_plotter import load_results, ts2xy

env_name = 'HalfCheetahBulletEnv-v0'
StartFresh = True
DoTraining = True
DoVideo = True

def main():
  # Create the callback: check every 1000 steps
  log_dir = 'new_log'
  num_cpu = 1
  model_stats_path = os.path.join(log_dir, "sac_" + env_name)
  env_stats_path = os.path.join(log_dir, 'sac_LR001.pkl')
  tb_log = 'tb_log'
  itr = 1
  videoName = 'M_timesteps_sac'
  tb_log_name = videoName

  if(StartFresh):
        env = DummyVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(num_cpu)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        env.reset()
        
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=tb_log)

  else:
      env = DummyVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(num_cpu)])
      env = VecNormalize.load(env_stats_path, env)
      env.reset()

      model = SAC.load(model_stats_path, tensorboard_log=tb_log)
      model.set_env(env)

  if(DoTraining):
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=1000000,
            tb_log_name=tb_log_name, 
            reset_num_timesteps=False) #, callback=callback, =TensorboardCallback()

    # Don't forget to save the VecNormalize statistics when saving the agent
    model.save(model_stats_path)
    env.save(env_stats_path)
    
  if(DoVideo):
    eval_env = make_vec_env(env_name, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env.reset()
    # mean_reward, std_reward = evaluate_policy(model, eval_env)
    # print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    record_video(env_name, model, video_length=2000, prefix='ppo_'+ env_name + videoName)

def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make(env_id)])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()

  

class SaveOnBestTrainingRewardCallback(BaseCallback):
    '''
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    '''
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_save_path = os.path.join(log_dir, 'best_model')
        self.current_save_path = os.path.join(log_dir, 'model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.best_save_path is not None:
            os.makedirs(self.best_save_path, exist_ok=True)
        if self.current_save_path is not None:
            os.makedirs(self.current_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f'Num timesteps: {self.num_timesteps}')
                    print(f'Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}')
                # Saving current model just for fun...
                if self.verbose > 0:
                    print(f'Saving current to {self.current_save_path + self.num_timesteps}.zip')
                self.model.save(self.current_save_path + self.num_timesteps)
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f'Saving new best model to {self.best_save_path + self.num_timesteps}.zip')
                    self.model.save(self.best_save_path + self.num_timesteps)

        return True

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
        env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    main()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True
