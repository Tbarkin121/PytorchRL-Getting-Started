


TIMESTEPS = 1000000
sched_LR = LinearSchedule(TIMESTEPS, 0.005, 0.00001)

model = PPO2(policy ='CustomPolicy', 
                env = env, 
                verbose = 1, 
                vf_coef = 1.0, 
                noptepochs = 5, 
                ent_coef = 0.005, 
                learning_rate = sched_LR.value,
                tensorboard_log = tensorboard_log_location,
                n_steps = 8192, 
                nminibatches = 128)

model.learn(total_timesteps = TIMESTEPS)