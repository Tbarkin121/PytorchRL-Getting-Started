import gym
import numpy as np
import math
import pybullet as p
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt


class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()
        
        EnGUI = False
        if(EnGUI):
            self.client = p.connect(p.GUI)
            self.angle = p.addUserDebugParameter('Steering', -0.6, 0.6, 0)
            self.throttle = p.addUserDebugParameter('Throttle', -1, 1, 0)
        else:
            self.client = p.connect(p.DIRECT)
        
        self.user_angle = None
        self.user_throttle = None

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.goalObj = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

        self.max_step_count = 2000
        self.current_step_count = 0
        
    def step(self, action):
        # Feed action to the car and get observation of car's state
        # action = self.get_user_action()
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        # reward = self.prev_dist_to_goal - dist_to_goal

        self.prev_dist_to_goal = dist_to_goal

        # Done by running off boundaries
        if (car_ob[0] >= 20 or car_ob[0] <= -20 or
                car_ob[1] >= 20 or car_ob[1] <= -20):
            self.done = True
            reward = -10

        # Done by reaching goal
        elif dist_to_goal < 1.5:
            reward = 50
            self.reset_goal()


        ob = np.array(car_ob + self.goal, dtype=np.float32)

        reward -= 0.01

        if(self.current_step_count > self.max_step_count):
            reward = 0
            self.done = True
        self.current_step_count += 1
        
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self.client)
        self.car = Car(self.client)

        # Set the goal to a random target
        self.reset_goal()
        self.done = False

        # # Visual element of the goal
        # self.goalObj = Goal(self.client, self.goal)

        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        
        self.current_step_count = 0

        return np.array(car_ob + self.goal, dtype=np.float32)

    def reset_goal(self):
        # Set the goal to a random target
        x = (self.np_random.uniform(2, 10) if self.np_random.randint(2) else
             self.np_random.uniform(-2, -10))
        y = (self.np_random.uniform(2, 10) if self.np_random.randint(2) else
             self.np_random.uniform(-2, -10))
        self.goal = (x, y)
        # Visual element of the goal
        if(p.getNumBodies()==3):
            p.removeBody(2) # I hope this removes the goal before adding a new one... need a better way of doing this though
        Goal(self.client, self.goal)

        # print("p.getNumBodies : {}".format(p.getNumBodies()))
        # for i in range(p.getNumBodies()):
        #     print("p.getBodyInfo(i) : {}".format(p.getBodyInfo(i)))


    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def get_user_action(self):
        self.user_angle = p.readUserDebugParameter(self.angle)
        self.user_throttle = p.readUserDebugParameter(self.throttle)
        return ([self.user_throttle , self.user_angle])

    def close(self):
        p.disconnect(self.client)
