import time
import numpy as np
import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
import transforms3d
import mujoco
import mujoco.viewer


def set_camera(viewer):
    cam = viewer.cam
    cam.distance = 3.0        # 相机距离目标点的距离
    cam.azimuth = 180         # 水平旋转角度（0-360度）
    cam.elevation = -30       # 俯仰角（上下视角）
    cam.lookat[:] = [0, 0, 0.5] # 相机注视点坐标 (x,y,z)

def view_pos():
    model = mujoco.MjModel.from_xml_path('model/booster_t1/T1_Serial_simulation.xml')
    data = mujoco.MjData(model)

    # 设置关节位置
    #joint_name, joint_angle = 'Head_pitch', -18 # 47, -18
    #joint_name, joint_angle = 'AAHead_yaw', -58 # 58, -58
    #joint_name, joint_angle = 'Left_Shoulder_Pitch', -188 # 68,-188
    #joint_name, joint_angle = 'Left_Shoulder_Roll', 88 # 88,-94
    #joint_name, joint_angle = 'Left_Elbow_Pitch', -128 # 128, -128
    #joint_name, joint_angle = 'Left_Elbow_Yaw', -120 # 2, -120
    #joint_name, joint_angle = 'Waist', -58 # 58, -58
    #joint_name, joint_angle = 'Left_Hip_Pitch', -118 # 118，-118
    #joint_name, joint_angle = 'Left_Hip_Roll', -21 # 88，-21
    #joint_name, joint_angle = 'Left_Hip_Yaw', -58 # 58, -58
    #joint_name, joint_angle = 'Left_Knee_Pitch', 0 # 123, 0
    #joint_name, joint_angle = 'Left_Ankle_Pitch', -23 # 49，-23
    joint_name, joint_angle = 'Left_Ankle_Roll', -24 # 45，-24

    # 获取关节的 qpos 地址
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_addr] = joint_angle/180*np.pi
    print(qpos_addr)

    print('qpos:', len(data.qpos), data.qpos)
    mujoco.mj_forward(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
'''
def view_ik_pos():
    model = mujoco.MjModel.from_xml_path('T1_Serial_simulation.xml')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    target = [0.3, 0, 1.0]
    base = data.body('Trunk').xpos
    for i in range(3): target[i] -= base[i]
    print(target)

    hand_link = np.linalg.norm(2*np.array([-0.000108, 0.109573, 0.000591]))
    print(f'hand_link: {hand_link:.3f}')

    left_arm_chain = ikpy.chain.Chain.from_urdf_file(
        'T1_Serial.urdf',
        base_elements=['Trunk', 'Left_Shoulder_Pitch'],
        last_link_vector=[0, hand_link, 0],
        active_links_mask=[False, True, True, True, True, False]
    )

    ik = left_arm_chain.inverse_kinematics(target)
    data.qpos[9:13] = ik[1:-1]
    print(ik)

    print('qpos:', len(data.qpos), data.qpos)
    mujoco.mj_forward(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
'''

def view_ik_pos_pd_control():
    model = mujoco.MjModel.from_xml_path('T1_Serial_simulation.xml')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 目标位置和逆运动学计算（同上）
    target = [0.3, 0, 1.0]
    base = data.body('Trunk').xpos
    for i in range(3): 
        target[i] -= base[i]

    hand_link = np.linalg.norm(2*np.array([-0.000108, 0.109573, 0.000591]))
    
    left_arm_chain = ikpy.chain.Chain.from_urdf_file(
        'T1_Serial.urdf',
        base_elements=['Trunk', 'Left_Shoulder_Pitch'],
        last_link_vector=[0, hand_link, 0],
        active_links_mask=[False, True, True, True, True, False]
    )

    target_ik = left_arm_chain.inverse_kinematics(target)
    target_q = target_ik[1:-1]
    
    # PD控制参数
    Kp = np.array([500, 500, 500, 500])  # 比例增益
    Kd = np.array([50, 50, 50, 50])      # 微分增益
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_camera(viewer)
        
        while viewer.is_running():
            # 获取当前关节状态
            current_q = data.qpos[9:13]
            current_qvel = data.qvel[8:12]  # 注意qvel索引可能不同
            
            # 计算误差
            position_error = target_q - current_q
            velocity_error = -current_qvel
            
            # PD控制器
            torque = Kp * position_error + Kd * velocity_error
            
            # 应用控制力矩（需要确认控制器索引）
            if hasattr(data, 'ctrl') and len(data.ctrl) > 12:
                data.ctrl[8:12] = torque  # 根据实际模型调整索引
            
            # 更新仿真
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

def test_ikpy():
    #base_elements = ['AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist']
    #ikpy.urdf.URDF.get_chain_from_joints('model/booster_t1/T1_Serial.urdf', )
    #return
    chain = ikpy.chain.Chain.from_urdf_file('model/booster_t1/T1_Serial.urdf', base_elements=['Trunk'])
    print(chain)
    return
    position = [-0.2, 0.5, 0.2]
    orientation = transforms3d.euler.euler2mat(0, 0, np.pi/2)
    #ref_pos = [0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]

    ik = chain.inverse_kinematics(position, orientation, 'all')#, initial_position=ref_pos)
    
    fig, ax = plot_utils.init_3d_figure()
    chain.plot(ik, ax)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.show()

def test_mujoco():
    #model = mujoco.MjModel.from_xml_path('model/h1_2_description/h1_2.xml')
    model = mujoco.MjModel.from_xml_path('model/booster_t1/T1_Serial.xml')
    #model = mujoco.MjModel.from_xml_path('model/booster_t1/T1_locomotion.xml')
    #model = mujoco.MjModel.from_xml_path('model/booster_t1/T1_7_dof_arm_serial_with_head_arm.xml')

    data = mujoco.MjData(model)
    
    print('data.ctrl:', len(data.ctrl))

    # 2: 左肩前摆(68,-188), 3: 左肩侧摆(88,-94)
    #data.ctrl[2] = 68
    #data.ctrl[3] = 88
    
    print(data.ctrl)
    data.qpos[10] = 88/180*np.pi  # [0. 0. 0.68 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #qpos = data.qpos
    print(data.xpos.shape)
    print('qpos:', len(data.qpos), data.qpos)
    #return
    mujoco.mj_forward(model, data)
    
    #option = mujoco.MjvOption()
    #option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()

    '''
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            #mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)
            #print(len(data.qpos), len(data.qvel), len(data.qacc), len(data.ctrl))
            data.xpos[:] = 0
            #data.qpos[:] = qpos  # 描述关节当前位置的变量
            data.qvel[:] = 0  # 描述关节当前速度的变量
            data.qacc[:] = 0  # 描述关节当前加速度的变量
            #print(data.qpos)
    '''

if __name__=='__main__':
    #view_pos()
    # view_ik_pos()
    view_ik_pos_pd_control()
    #test_ikpy()
    #test_mujoco()
