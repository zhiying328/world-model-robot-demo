import pybullet as p
import numpy as np

def render_frame():
    width, height, rgba, depth, seg = p.getCameraImage(
        width=256,
        height=256,
        viewMatrix=p.computeViewMatrix(
            cameraEyePosition=[2, 2, 2],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        ),
        projectionMatrix=p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=10
        )
    )

    # numpy
    img = np.array(rgba, dtype=np.uint8)

    # reshape 成图像
    img = img.reshape((height, width, 4))

    # 取 RGB
    return img[:, :, :3]