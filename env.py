import pybullet as p
import pybullet_data
import numpy as np


class DroneEnv:
    def __init__(self, gui=False):
        if gui:
            self._client = p.connect(p.GUI)
        else:
            self._client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self._drone_id = None
        self._state = np.zeros(6)
        self._dt = 0.05

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Floor
        p.loadURDF("plane.urdf")

        # Drone - try quadrotor.urdf, fallback to a visible blue box
        try:
            self._drone_id = p.loadURDF("quadrotor.urdf", [0, 0, 1])
        except Exception:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.1])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.1],
                                      rgbaColor=[0.1, 0.4, 1.0, 1])
            self._drone_id = p.createMultiBody(0.5, col, vis, [0, 0, 1])

        self._state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        obs = self._render()
        return self._state.copy(), obs

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        x, y, z, vx, vy, vz = self._state

        # Gravity-compensated kinematic update (velocity accumulates)
        ax, ay, az = action[0], action[1], action[2]
        vx += ax * self._dt
        vy += ay * self._dt
        vz += az * self._dt

        vx = np.clip(vx, -2.0, 2.0)
        vy = np.clip(vy, -2.0, 2.0)
        vz = np.clip(vz, -1.0, 1.0)

        x += vx * self._dt
        y += vy * self._dt
        z += vz * self._dt
        z = max(z, 0.1)

        self._state = np.array([x, y, z, vx, vy, vz])
        p.resetBasePositionAndOrientation(self._drone_id, [x, y, z], [0, 0, 0, 1])

        obs = self._render()
        return self._state.copy(), obs

    def render_at_pos(self, pos):
        """Render scene with drone moved to an arbitrary position (for world model visualization)."""
        orig_pos = self._state[:3].copy()
        p.resetBasePositionAndOrientation(self._drone_id, pos, [0, 0, 0, 1])
        # Temporarily update state for camera tracking
        orig_state = self._state.copy()
        self._state[:3] = pos
        obs = self._render()
        # Restore
        self._state = orig_state
        p.resetBasePositionAndOrientation(self._drone_id, orig_pos, [0, 0, 0, 1])
        return obs

    def _render(self):
        x, y, z = self._state[:3]
        # Camera close to drone: offset 1m back, 0.8m up
        eye = [x + 1.0, y + 1.0, z + 0.8]
        target = [x, y, z]

        width, height, rgba, _, _ = p.getCameraImage(
            width=128,
            height=128,
            viewMatrix=p.computeViewMatrix(eye, target, [0, 0, 1]),
            projectionMatrix=p.computeProjectionMatrixFOV(70, 1, 0.1, 20),
        )
        img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
        return img[:, :, :3]

    def close(self):
        p.disconnect()
