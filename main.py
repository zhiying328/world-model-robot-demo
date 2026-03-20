import os
import json
import numpy as np
import imageio
from math import pi, cos, sin
from PIL import Image, ImageDraw

from env import DroneEnv
from world_model import WorldModel

os.makedirs("output", exist_ok=True)

T = 80


def sinusoidal_flight_path(t):
    theta = 2 * pi * t / T
    return np.array(
        [0.4 * cos(theta), 0.4 * sin(theta), 0.15 * sin(2 * theta)],
        dtype=np.float32,
    )


def annotate_frame(img_arr, lines):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    y = 4
    for line in lines:
        draw.text((4, y), line, fill=(255, 255, 255))
        y += 12
    return np.array(img)


def make_composite(real_obs, imagined_obs, state, action, next_state):
    real_lines = [
        "REAL ENV",
        f"pos {state[0]:.2f} {state[1]:.2f} {state[2]:.2f}",
        f"act {action[0]:.2f} {action[1]:.2f} {action[2]:.2f}",
        f"vel {state[3]:.2f} {state[4]:.2f} {state[5]:.2f}",
    ]
    imag_lines = [
        "WORLD MODEL",
        f"nxt {next_state[0]:.2f} {next_state[1]:.2f} {next_state[2]:.2f}",
        "(RSSM+Decoder)",
    ]
    real_ann = annotate_frame(real_obs, real_lines)
    imag_ann = annotate_frame(imagined_obs, imag_lines)
    composite = np.concatenate([real_ann, imag_ann], axis=1)  # [128, 256, 3]
    return composite


env = DroneEnv(gui=False)
wm = WorldModel()

state, obs = env.reset()
wm.reset()

frames = []
trajectory = []

for t in range(T):
    action = sinusoidal_flight_path(t)

    z_next, imagined_obs = wm.predict(obs, action)
    next_state, next_obs = env.step(action)

    composite = make_composite(obs, imagined_obs, state, action, next_state)
    frames.append(composite)

    entry = {
        "t": t,
        "state": state.tolist(),
        "action": action.tolist(),
        "next_state": next_state.tolist(),
    }
    trajectory.append(entry)

    print(
        f"t={t:02d} | state=[{state[0]:.2f},{state[1]:.2f},{state[2]:.2f}]"
        f" | action=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f}]"
        f" | next_state=[{next_state[0]:.2f},{next_state[1]:.2f},{next_state[2]:.2f}]"
    )

    obs = next_obs
    state = next_state

env.close()

imageio.mimsave("output/drone_dreamer.gif", frames, fps=15)

with open("output/trajectory.json", "w") as f:
    json.dump(trajectory, f, indent=2)

print("Done! output/drone_dreamer.gif and output/trajectory.json written.")
