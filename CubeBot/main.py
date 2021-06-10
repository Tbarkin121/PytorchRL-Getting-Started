import pybullet as p
from time import sleep
# Can alternatively pass in p.DIRECT 
client = p.connect(p.GUI)
p.setGravity(0, 0, -0.01, physicsClientId=client) 

import pybullet_data
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("simple_driving/resources/simpleplane.urdf")
# carId = p.loadURDF("racecar/racecar.urdf", basePosition=[0,0,0.2])
# carId = p.loadURDF("husky/husky.urdf", basePosition=[0,0,0.2])
# carId = p.loadURDF("bicycle/bike.urdf", basePosition=[0,0,2.2])
# carId = p.loadURDF("laikago/laikago.urdf", basePosition=[0,0,2.2])
botId = p.loadURDF('simple_driving/resources/cubebot.urdf', basePosition=[0,0,1.2])

# position, orientation = p.getBasePositionAndOrientation(carId)
# for _ in range(2500000): 
#     p.stepSimulation()


number_of_joints = p.getNumJoints(botId)
for joint_number in range(number_of_joints):
    info = p.getJointInfo(botId, joint_number)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(info[0], ": ", info[1])
    print('!!!!!!!!!!!!!!!!!!!!!!!!!')

throttle_1 = p.addUserDebugParameter('Throttle', -10, 10, 0)
throttle_2 = p.addUserDebugParameter('Throttle', -10, 10, 0)

while True:
    user_throttle_1 = p.readUserDebugParameter(throttle_1)
    user_throttle_2 = p.readUserDebugParameter(throttle_2)
    
    p.setJointMotorControl2(botId, 0,
                            p.VELOCITY_CONTROL,
                            targetVelocity=user_throttle_1)
    
    p.setJointMotorControl2(botId, 1,
                            p.VELOCITY_CONTROL,
                            targetVelocity=user_throttle_2)

    p.stepSimulation()