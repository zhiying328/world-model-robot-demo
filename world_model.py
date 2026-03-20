import numpy as np

def predict_next_state(state, action):
    """
    state: [x, y, z, vx, vy, vz]
    action: [ax, ay, az]
    """

    x, y, z, vx, vy, vz = state
    ax, ay, az = action

    dt = 0.1

    # 简单物理更新
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    vz_new = vz + az * dt

    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    z_new = max(0.1, z + vz_new * dt)

    return np.array([x_new, y_new, z_new, vx_new, vy_new, vz_new])