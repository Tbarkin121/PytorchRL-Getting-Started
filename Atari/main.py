from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    print("Did you even try?")
    eval_env = make_atari_env(env_id, n_envs=nEnv, seed=0)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecVideoRecorder(eval_env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="trained-agent-{}".format(env_id))

    obs = eval_env.reset()                           
    for _ in range(video_length):
        action, _states  = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        eval_env.render()

    # Close the video recorder
    eval_env.close()

# Stack 4 frames
env_id  = 'PongNoFrameskip-v4'
video_folder = 'logs/videos/'
video_length = 1000
nEnv = 16
startFresh = False
if (startFresh):
    env = make_atari_env(env_id, n_envs=nEnv, seed=0)
    env = VecFrameStack(env, n_stack=4)
    env.reset()
    model = A2C('CnnPolicy', env, verbose=1, tensorboard_log="./a2c_pong_tensorboard/")
    # model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_pong_tensorboard/")
    model.learn(total_timesteps=1000000)
    model.save("a2c_pong_{}".format(model.num_timesteps))
    # model.save("ppo_pong_{}".format(model.num_timesteps))
    record_video(env_id, model, video_length=500, prefix='ac2_'+env_id, video_folder='videos/')
    
else:
    env = make_atari_env(env_id, n_envs=nEnv, seed=0)
    env = VecFrameStack(env, n_stack=4)
    env.reset()
    trained_model = A2C.load("a2c_pong_6000000", verbose=1)
    # trained_model = PPO.load("ppo_pong_1015808", verbose=1)
    trained_model.set_env(env)
    # trained_model.learn(total_timesteps=1000000, reset_num_timesteps=False)
    trained_model.save("a2c_pong_{}".format(trained_model.num_timesteps))
    # trained_model.save("ppo_pong_{}".format(trained_model.num_timesteps))
    record_video(env_id, trained_model, video_length=500, prefix='ac2_'+env_id, video_folder='videos/')





