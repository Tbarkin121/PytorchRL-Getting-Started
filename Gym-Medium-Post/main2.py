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
    num_cpu = 16
    model_stats_path = os.path.join(log_dir, 'Model_' + tb_log_name)
    env_stats_path = os.path.join(log_dir, 'Env_' + tb_log_name)
    checkpoint_path = os.path.join(log_dir, 'saved_models')
    best_path = os.path.join(log_dir, 'best_models')
    load_path = os.path.join(best_path, 'best_model.zip')
    video_path = os.path.join(log_dir, videoName)
    tb_log = os.path.join(log_dir, 'tb_log')
    eval_freq = 50000/num_cpu #The num_cpu seemed to factor in. 1000 = 16000 for 16 cpu
    vid_freq = 10000 #Well... This doesn't follow the pattern as above... ok whatever
    total_timesteps = 3000000
    # Some Controls to what happens...
    StartFresh = True
    DoTraining = True
    DoVideo = False


    if(StartFresh):
        # Create Environment
        env = SubprocVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(num_cpu)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        env.reset()
        # Separate evaluation env
        eval_env = SubprocVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(1)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        eval_env.reset()
        # Create Model
        # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=tb_log, device="auto")
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])])

        model = PPO('MlpPolicy', 
            env, 
            learning_rate = 3e-5,
            n_steps=512,
            batch_size=128,
            n_epochs=20,
            gamma=0.99,
            gae_lambda = 0.9,
            clip_range = 0.4,
            vf_coef = 0.5,
            use_sde = True,
            sde_sample_freq = 4,
            policy_kwargs = policy_kwargs, 
            verbose=1, 
            tensorboard_log=tb_log,
            device="auto")


    else:
        print('duh')
        # tmp_test_name = 'SAC-Continued'
        # tb_log_name = tmp_test_name + '_' + env_name
        # tmp_log_dir = os.path.join('log', tmp_test_name)
        # tmp_model_stats_path = os.path.join(tmp_log_dir, 'Model_' + tb_log_name)
        # tmp_env_stats_path = os.path.join(tmp_log_dir, 'Env_' + tb_log_name)
        # tmp_best_path = os.path.join(tmp_log_dir, 'saved_models')
        # tmp_load_path = os.path.join(tmp_best_path, 'rl_model_3900000_steps')
        # # Load Enironment
        # env = DummyVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(num_cpu)])
        # env = VecNormalize.load(tmp_env_stats_path, env)
        # env.reset()
        # # Separate evaluation env
        # eval_env = DummyVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(num_cpu)])
        # eval_env = VecNormalize.load(tmp_env_stats_path, eval_env)
        # eval_env.reset()
        # # Load Model
        # # model = SAC.load(model_stats_path, tensorboard_log=tb_log)
        # model = SAC.load(tmp_load_path, tensorboard_log=tb_log, learning_rate=1e-6)
        # # model.learning_rate = 1e-5
        # model.set_env(env)

    if(DoTraining):
        checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=checkpoint_path)
        # Use deterministic actions for evaluation
        eval_callback = EvalCallback(eval_env, best_model_save_path=best_path,
                                    log_path=best_path, eval_freq=eval_freq,
                                    deterministic=True, render=False)
        # Video Update Callback 
        record_callback = RecordVideo(env_name, videoName=videoName, videoPath=video_path, verbose=1)
        envSave_callback = SaveEnvVariable(env, model, env_stats_path, model_stats_path)
        nStep_callback_list = CallbackList([record_callback, envSave_callback])
        # nStep_callback_list = CallbackList([envSave_callback])
        vid_callback = EveryNTimesteps(n_steps=vid_freq, callback=nStep_callback_list)
        
        # Create the callback list
        # callbacks = CallbackList([checkpoint_callback, eval_callback, vid_callback])
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        print(tb_log_name)
        model.learn(total_timesteps=total_timesteps,
            tb_log_name=tb_log_name, 
            reset_num_timesteps=False,
            callback=callbacks)

        # Don't forget to save the VecNormalize statistics when saving the agent
        model.save(model_stats_path)
        env.save(env_stats_path)

    if(DoVideo):
        record_video(env_name, env, model, videoLength=1000, prefix='best' + videoName, videoPath=video_path, log_dir=log_dir)
    
    

class RecordVideo(BaseCallback):
    def __init__(self, env_name, videoName='video', videoPath='videos/', videoLength = 1000, verbose=1):
        super(RecordVideo, self).__init__(verbose)
        self.env_name = env_name
        self.videoName = videoName
        self.videoPath = videoPath
        self.videoLength = videoLength
        
        #self.num_timesteps
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.videoPath is not None:
            os.makedirs(self.videoPath, exist_ok=True)

    def _on_step(self) -> bool:
        print('Record A Video')
        _videoName = self.videoName + '_' + str(self.model.num_timesteps)
        print(_videoName)
        record_video(self.env_name, self.training_env, self.model, videoLength=self.videoLength, prefix=_videoName, videoPath=self.videoPath)

class SaveEnvVariable(BaseCallback):
    def __init__(self, env, model, _env_stats_path, _model_stats_path, verbose=1):
        super(SaveEnvVariable, self).__init__(verbose)
        self.env = env
        self.model = model
        self.env_stats_path = _env_stats_path
        self.model_stats_path = _model_stats_path
        
    
    def _on_step(self) -> bool:
        print('Save Env Variables')
        self.model.save(model_stats_path)
        self.env.save(env_stats_path)

def record_video(env_name, train_env, model, videoLength=500, prefix='', videoPath='videos/', log_dir=''):
    print('record_video function')
    # Wrap the env in a Vec Video Recorder 
    local_eval_env = SubprocVecEnv([make_env(env_name, i, log_dir=log_dir) for i in range(1)])
    local_eval_env = VecNormalize(local_eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    sync_envs_normalization(train_env, local_eval_env)
    local_eval_env = VecVideoRecorder(local_eval_env, video_folder=videoPath,
                              record_video_trigger=lambda step: step == 0, video_length=videoLength,
                              name_prefix=prefix)
    obs = local_eval_env.reset()
    for _ in range(videoLength):
        action, _ = model.predict(obs)
        obs, _, _, _ = local_eval_env.step(action)

    # Close the video recorder
    local_eval_env.close()

# To Do: Check out this log_dir variable. Maybe it would help stop the issue I was having with the vectorized environments crashing
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

if __name__ == '__main__':
    main()