import numpy as np


class WorldModel:
    """
    Simplified kinematic world model.
    Real env accumulates velocity across steps; this model treats each step
    independently (no velocity memory), causing visible drift over time.
    """

    def __init__(self, dt=0.05):
        self._dt = dt

    def reset(self):
        pass

    def predict(self, state, action):
        """
        state:  np.array [x, y, z, vx, vy, vz]
        action: np.array [ax, ay, az]
        Returns predicted next_state [x, y, z, vx, vy, vz]
        """
        x, y, z = state[0], state[1], state[2]
        ax, ay, az = action[0], action[1], action[2]

        # Simplified: ignores velocity history, only uses action directly
        x_next = x + ax * self._dt
        y_next = y + ay * self._dt
        z_next = max(z + az * self._dt, 0.1)

        return np.array([x_next, y_next, z_next, ax * self._dt, ay * self._dt, az * self._dt])
