import pybullet as p
import pybullet_data

# Can alternatively pass in p.DIRECT 
client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client) 

p.setAdditionalSearchPath(pybullet_data.getDataPath())

planeId = p.loadURDF("plane.urdf")
carId = p.loadURDF("racecar/racecar.urdf", basePosition=[0,0,0.2])

for _ in range(100000): 
    pos, ori = p.getBasePositionAndOrientation(carId)
    p.applyExternalForce(carId, 0, [29, 0, 0], pos, p.WORLD_FRAME)
    p.stepSimulation()