import pybullet as p
import pybullet_data

def create_env(gui=False):
    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    plane = p.loadURDF("plane.urdf")
    drone = p.loadURDF("sphere2.urdf", [0, 0, 1])  # 用球当无人机

    return drone