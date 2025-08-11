# # import mujoco #注意不是mujuco_py
# # print(mujoco.__version__)
# import mujoco
# import mujoco.viewer as viewer
# import sys,os
#
# viewer.launch_from_path("./h1_2_description/robot.xml")
import mujoco
import mujoco.viewer
import numpy as np
import time
import numpy as np
from scipy.optimize import root

A0 = 0.07106
A1 = 0.275437
A2 = 0.232857
A3 = 0.054
G = 0.2618
Z = 0.0896
Y = 0.2868
X = 0.3346
num=19
model = mujoco.MjModel.from_xml_path("./h1_2_description/robot.xml")
data = mujoco.MjData(model)

Kp = np.array([500 for i in range(num)])
Kd = np.array([20 for j in range(num)])
# desired_qpos = np.zeros(num)
#
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     last_qvel = np.zeros(num)
#     while viewer.is_running():
#         start = time.time()
#
#         current_qpos = np.zeros(num)#data.qpos[:num]
#         current_qvel = np.zeros(num)#data.qvel[:num]
#
#         error = desired_qpos - current_qpos
#         d_error = -current_qvel
#
#         torque = Kp * error + Kd * d_error
#        # torque = Kp * error
#         data.ctrl[:num] = torque  # 发送力矩控制信号
#
#         mujoco.mj_step(model, data)
#
#         viewer.sync()
#
#         # 200Hz
#         time.sleep(max(0, 0.005 - (time.time() - start)))

desired_amplitude = np.pi / 8

frequency = 1

def equations(vars):
    x0, x1, x2 = vars
    eq1 = (-A1 * np.cos(x0) + A2 * np.sin(x0 - x2)) * np.cos(x1) + A0 * np.sin(G) - Z
    eq2 = (A1 * np.sin(x0) + A2 * np.cos(x0 - x2) + A3) - Y
    eq3 = (A1 * np.cos(x0) - A2 * np.sin(x0 - x2)) * np.sin(x1) + A0 * np.cos(G) - X

    return [eq1, eq2, eq3]

# 初始猜测值
initial_guess = [0.5, 0.5, 0.5]
solution = root(equations, initial_guess)

if solution.success:
    x0_num, x1_num, x2_num = solution.x
    print(x0_num)
    x0_num=np.arctan(np.tan(x0_num)*np.cos(x1_num-G))
    print(x0_num)
    j3_num = x0_num - x2_num
    if -3.14 <= x0_num <= 1.57 and -0.38 <= x1_num <= 3.4 and -0.471 <= x2_num <= 0.349 and -1.012 <= j3_num <= 1.012:
        print(f"解为: x0 = {x0_num}, x1 = {x1_num}, x2 = {x2_num}, x3 = {j3_num}")
    else:
        print("超过关节限制")
else:
    print("方程组无解")

site_name_2 = "left_wrist_yaw_site"  # 替换为你的site名称   torso_link
#site_name_2="L_thumb_proximal_yaw_site"
site_name_1 = "left_shoulder_pitch_site"
#site_name_1 = "left_wrist_roll_site"
site_id_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name_1)
site_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name_2)
with mujoco.viewer.launch_passive(model, data) as viewer:
    last_qvel = np.zeros(num)
    while viewer.is_running():
        start = time.time()

        current_qpos = data.qpos[:num]
        current_qvel = data.qvel[:num]

        t = time.time()
        #desired_qpos = desired_amplitude * np.sin(2 * np.pi * frequency * t) * np.ones(num)
        desired_qpos =[0]*12+[-x0_num,x1_num,0,x2_num,np.pi/2,0,-j3_num] + [0]*(num-19)
        #desired_qpos = [0] * 12 + [0,0,0,0,0,0,0] + [0] * (num - 19)
        error = desired_qpos - current_qpos
        d_error = -current_qvel

        torque = Kp * error + Kd * d_error
        data.ctrl[:num] = torque

        mujoco.mj_step(model, data)
        # 获取site的全局坐标
        site_pos_1 = data.site_xpos[site_id_1]
        site_pos_2 = data.site_xpos[site_id_2]
        print(f"Site {site_name_1} 坐标: {site_pos_1}")
        print(f"Site {site_name_2} 坐标: {site_pos_2}")
        print(error)
        viewer.sync()

        # 200Hz 控制频率
