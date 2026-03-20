import numpy as np
import imageio
import pybullet as p

from env import create_env
from world_model import predict_next_state
from render import render_frame

# ========================
# 初始化环境
# ========================
drone = create_env(gui=False)

# 初始状态
state = np.array([0, 0, 1, 0, 0, 0])

frames = []

# 主循环
for t in range(80):

    # 随机动作（控制）
    action = np.random.uniform(-1, 1, size=3)

    # 世界模型预测
    next_state = predict_next_state(state, action)

    # 更新仿真位置（用于渲染）
    p.resetBasePositionAndOrientation(
        drone,
        next_state[:3],
        [0, 0, 0, 1]
    )

    # 渲染
    frame = render_frame()
    frames.append(frame)

    # 打印数据（你作业要用）
    print("state:", state)
    print("action:", action)
    print("next_state:", next_state)
    print("------")

    # 更新状态
    state = next_state

# 保存视频
imageio.mimsave("output/drone_demo.gif", frames, fps=10)

print("完成！输出在 output/drone_demo.gif")