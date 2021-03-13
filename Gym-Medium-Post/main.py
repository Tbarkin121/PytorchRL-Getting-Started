import gym
import torch
from agent import TRPOAgent
import simple_driving
import time
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_checker import check_env
import numpy as np
from typing import Callable

env_id = 'SimpleDriving-v0'
_log_dir = 'log\monitor_simpledriving_vecNormalized_128x3_2\\'
_stats_path = _log_dir + 'vec_normalize_2.pkl'
tb_log = _log_dir + 'SubprocVecEnv_8env_128x3'

def record_video(env_id, model, video_length=500, prefix='', video_folder='videos'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
        """
    eval_env = DummyVecEnv([make_env(env_id, i, log_dir=_log_dir) for i in range(1)])
    # eval_env = gym.make(env_id)
    val_env = VecNormalize.load(_log_dir + 'vec_normalize_5734400.pkl', eval_env)

    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder='tmp',
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)
                            
    obs = eval_env.reset()
    for i in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()

def main():
    # nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
    #                          torch.nn.Linear(64, 2))
    
    os.makedirs(_log_dir, exist_ok=True)

    DoTraining = True
    StartFresh = True
    num_cpu = 8
    if(DoTraining):
        
        
        # This doesn't work but it might have something to do with how the environment is written
        # num_cpu = 1 
        # env = make_vec_env(env_id, n_envs=num_cpu, monitor_dir=_log_dir) # make_vec_env contains Monitor
        
        # Create the callback: check every 1000 steps
        # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=_log_dir)
        
        if(StartFresh):
            env = SubprocVecEnv([make_env(env_id, i, log_dir=_log_dir) for i in range(num_cpu)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
            env.reset()
            policy_kwargs = {
                'net_arch':[128,128,128],
            }
            model = PPO('MlpPolicy', env, policy_kwargs = policy_kwargs, verbose=2, tensorboard_log=tb_log)
        else:
            env = SubprocVecEnv([make_env(env_id, i, log_dir=_log_dir) for i in range(num_cpu)])
            env = VecNormalize.load(_stats_path, env)
            env.reset()

            model = PPO.load('log\monitor_simpledriving_vecNormalized_128x3_2\PPO_4243456.mdl', tensorboard_log=tb_log)
            model.set_env(env)

        eval_env = gym.make(env_id) 
        # print('!!!!Checking Environment!!!!')
        # print(check_env(eval_env))

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f'mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}')
        for _ in range(50):
            model.learn(total_timesteps=100000, tb_log_name=env_id, reset_num_timesteps=False) #, callback=callback
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
            print(f'mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}')
            model.save(_log_dir + 'PPO_{}'.format(model.num_timesteps) + '.mdl')
            env.save(_log_dir + 'vec_normalize_{}'.format(model.num_timesteps) + '.pkl')

   
    if(not DoTraining):
        # eval_env = SubprocVecEnv([make_env(env_id, i, log_dir=_log_dir) for i in range(num_cpu)])
        # eval_env = VecNormalize.load(_log_dir + 'vec_normalize_5734400.pkl', eval_env)
        # eval_env = VecVideoRecorder(eval_env, video_folder='videos/',
        #                       record_video_trigger=lambda step: step == 0, video_length=500,
        #                       name_prefix='test')
        # eval_env.training = False
        # eval_env.norm_reward = False
        # eval_env.reset()
       
        eval_env = DummyVecEnv([make_env(env_id, i, log_dir=_log_dir) for i in range(1)])
        # eval_env = gym.make(env_id)
        eval_env = VecNormalize.load(_log_dir + 'vec_normalize_5734400.pkl', eval_env)
        
        model = PPO.load('log\monitor_simpledriving_vecNormalized_128x3\PPO_5734400.mdl', tensorboard_log=tb_log)
        model.set_env(eval_env)
        # record_video(env_id, model, video_length=500, prefix='ppo_'+env_id)
        # Start the video at step=0 and record 500 steps
        # eval_env = VecVideoRecorder(eval_env, video_folder='tmp',
        #                       record_video_trigger=lambda step: step == 0, video_length=500,
        #                       name_prefix='')

        obs = eval_env.reset()
        # for i in range(500):
        #     action, _ = model.predict(obs)
        #     obs, _, _, _ = eval_env.step(action)
        # eval_env.close()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)
            # eval_env.render()
            if done.any():
                # obs = eval_env.reset()
                # time.sleep(1/30)
                eval_env.close()
                break



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


# This doesn't seem to be threadsafe... running on multiple processors is casuing some sort of collition
# And DummyVecEnv isn't working right now so I cant try that out this second... 
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
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f'Saving new best model to {self.save_path}.zip')
                  self.model.save(self.save_path)

        return True

if __name__ == '__main__':
    main()


