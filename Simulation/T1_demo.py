import mujoco as mj
import mujoco.viewer
import numpy as np

# 加载模型
model = mj.MjModel.from_xml_path("T1_serial.xml")
data = mj.MjData(model)

# 设置初始姿态
mj.mj_resetData(model, data)
data.qpos[2] = 1.0  # 设置Z轴高度
data.qpos[7] = 0.1   # 左膝
data.qpos[14] = 0.1  # 右膝

# 保存目标位置
target_qpos = data.qpos[7:7+model.nu].copy()
mj.mj_forward(model, data)

# 启动可视化
with mujoco.viewer.launch_passive(model, data) as viewer:   
    while viewer.is_running():
        # 固定基座位置和姿态
        data.qpos[0:3] = [0, 0, 1.0]  # 固定位置
        data.qpos[3:7] = [1, 0, 0, 0]  # 固定姿态(四元数)
        data.qvel[0:6] = 0  # 清零基座速度
        
        # PD控制关节
        kp, kd = 100, 10
        data.ctrl[:] = kp * (target_qpos - data.qpos[7:7+model.nu]) - kd * data.qvel[6:6+model.nu]
        
        mj.mj_step(model, data)
        viewer.sync()