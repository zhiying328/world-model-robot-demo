import pybullet as p
import numpy as np

def render_frame():
    width, height, rgba, _, _ = p.getCameraImage(
        width=128,
        height=128,
        viewMatrix=p.computeViewMatrix([2,2,2],[0,0,0],[0,0,1]),
        projectionMatrix=p.computeProjectionMatrixFOV(60,1,0.1,10)
    )

    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
    return img[:, :, :3]