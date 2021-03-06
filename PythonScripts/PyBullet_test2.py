# import pybullet as p 
# from time import sleep
# p.connect(p.GUI)
# p.loadURDF("simplecar.urdf") 
# sleep(20)

# import pybullet as p
# p.connect(p.DIRECT)
# car = p.loadURDF('simplecar.urdf')
# number_of_joints = p.getNumJoints(car)
# for joint_number in range(number_of_joints):
#     info = p.getJointInfo(car, joint_number)
#     # print(info)
#     print(info[0], ": ", info[1])

import pybullet as p
from time import sleep

wheel_indices = [1, 3, 4, 5]
hinge_indices = [0, 2]

p.connect(p.GUI)
p.setGravity(0, 0, -10)
angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
throttle = p.addUserDebugParameter('Throttle', -20, 20, 0)
car = p.loadURDF('simplecar.urdf', [0, 0, 0.1])
plane = p.loadURDF('simpleplane.urdf')

while True:
    user_angle = p.readUserDebugParameter(angle)
    user_throttle = p.readUserDebugParameter(throttle)
    for joint_index in wheel_indices:
        p.setJointMotorControl2(car, joint_index,
                                p.VELOCITY_CONTROL,
                                targetVelocity=user_throttle)
    for joint_index in hinge_indices:
        p.setJointMotorControl2(car, joint_index,
                                p.POSITION_CONTROL, 
                                targetPosition=user_angle)
    p.stepSimulation()