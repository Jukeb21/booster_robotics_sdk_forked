#!/usr/bin/env python3
"""
使用 Robotics Toolbox Python 版本实现 URDF 导入和逆运动学求解
本示例演示如何加载 H1_2 机器人右臂的 URDF 模型并执行逆运动学计算
"""

import numpy as np
import os
import sys
from pathlib import Path

try:
    import roboticstoolbox as rtb
    from roboticstoolbox import DHRobot, RevoluteMDH
    from spatialmath import SE3
    import matplotlib.pyplot as plt
    print("成功导入 Robotics Toolbox 和相关库")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装以下依赖包:")
    print("pip install robotics-toolbox-python")
    print("pip install spatialmath-python")
    print("pip install matplotlib")
    sys.exit(1)


class H1_2_RightArmController:
    """H1_2 机器人右臂控制器"""
    
    def __init__(self):
        self.robot = None
        self.urdf_path = None
        self.setup_paths()
        
    def setup_paths(self):
        """设置URDF文件路径"""
        # 获取当前脚本所在目录
        current_dir = Path(__file__).parent.absolute()
        
        # 查找right_arm.urdf文件
        possible_paths = [
            current_dir.parent / "right_arm.urdf",  # 上级目录
            current_dir.parent.parent.parent / "h1_2_description" / "h1_2.urdf",  # h1_2_description目录
        ]
        
        for path in possible_paths:
            if path.exists():
                self.urdf_path = str(path)
                print(f"找到URDF文件: {self.urdf_path}")
                break
        
        if not self.urdf_path:
            print("未找到URDF文件，将使用DH参数手动创建机器人模型")

    def load_urdf_robot(self):
        """从URDF文件加载机器人模型"""
        if self.urdf_path and os.path.exists(self.urdf_path):
            try:
                # 尝试从URDF文件加载机器人
                print(f"正在从URDF文件加载机器人: {self.urdf_path}")
                self.robot = rtb.Robot.URDF(self.urdf_path)
                print(f"成功加载机器人模型: {self.robot.name}")
                print(f"机器人自由度: {self.robot.n}")

                # 修复：正确获取关节信息
                joint_names = []
                for i, link in enumerate(self.robot.links):
                    if hasattr(link, 'name') and link.name:
                        joint_names.append(link.name)
                    else:
                        joint_names.append(f"joint_{i}")
                print(f"连杆名称: {joint_names}")

                # 显示关节限制信息
                print("关节限制:")
                for i, link in enumerate(self.robot.links):
                    if hasattr(link, 'qlim') and link.qlim is not None:
                        qmin, qmax = link.qlim
                        print(f"  关节 {i + 1} ({joint_names[i]}): [{np.degrees(qmin):.1f}°, {np.degrees(qmax):.1f}°]")
                    else:
                        print(f"  关节 {i + 1} ({joint_names[i]}): 无限制")

                return True
            except Exception as e:
                print(f"从URDF加载失败: {e}")
                print("将使用DH参数手动创建机器人模型")
                return False
        return False
        
    def create_dh_robot(self):
        """使用DH参数手动创建H1_2右臂机器人模型"""
        print("使用DH参数创建H1_2右臂机器人模型...")
        
        # H1_2右臂的DH参数 (根据URDF文件中的关节配置估算)
        # 这些参数是基于URDF文件中的关节配置进行的估算
        dh_params = [
            # d, a, alpha, offset, qlim
            RevoluteMDH(d=0.0, a=0.0, alpha=np.pi/2, offset=0, qlim=[-3.14, 1.57]),      # 肩部俯仰
            RevoluteMDH(d=0.0, a=0.0, alpha=np.pi/2, offset=0, qlim=[-3.4, 0.38]),       # 肩部横滚
            RevoluteMDH(d=-0.22, a=0.0, alpha=np.pi/2, offset=0, qlim=[-1.8, 1.8]),      # 肩部偏航
            RevoluteMDH(d=0.0, a=0.0, alpha=np.pi/2, offset=0, qlim=[-3.14, 0]),         # 肘部
            RevoluteMDH(d=-0.22, a=0.0, alpha=np.pi/2, offset=0, qlim=[-1.8, 1.8]),      # 腕部俯仰
            RevoluteMDH(d=0.0, a=0.0, alpha=0, offset=0, qlim=[-1.8, 1.8]),              # 腕部横滚
        ]
        
        self.robot = DHRobot(
            dh_params,
            name='H1_2_RightArm',
            manufacturer='Unitree'
        )
        
        print(f"成功创建机器人模型: {self.robot.name}")
        print(f"机器人自由度: {self.robot.n}")
        return True
        
    def initialize_robot(self):
        """初始化机器人模型"""
        # 首先尝试从URDF加载
        if not self.load_urdf_robot():
            # 如果URDF加载失败，使用DH参数创建
            self.create_dh_robot()
            
        if self.robot is None:
            raise Exception("无法创建机器人模型")
            
        return True
        
    def demonstrate_forward_kinematics(self):
        """演示正运动学"""
        print("\n=== 正运动学演示 ===")
        
        # 定义一组关节角度
        q = np.array([0.1, -0.2, 0.3, -1.5, 0.1, 0.0])
        if len(q) > self.robot.n:
            q = q[:self.robot.n]
        elif len(q) < self.robot.n:
            q = np.concatenate([q, np.zeros(self.robot.n - len(q))])
            
        print(f"关节角度: {np.degrees(q)} (度)")
        
        # 计算正运动学
        T = self.robot.fkine(q)
        print(f"末端执行器位置: {T.t}")
        print(f"末端执行器姿态矩阵:\n{T.R}")
        
        return T
        
    def demonstrate_inverse_kinematics(self):
        """演示逆运动学"""
        print("\n=== 逆运动学演示 ===")
        
        # 定义目标位置和姿态
        target_position = [0.35, -0.22, 0.08]  # 目标位置 (x, y, z)
        target_orientation = np.eye(3)        # 目标姿态矩阵 (单位矩阵表示无旋转)
        
        # 创建目标变换矩阵
        T_target = SE3.Rt(target_orientation, target_position)
        print(f"目标位置: {target_position}")
        print(f"目标姿态矩阵:\n{target_orientation}")
        
        try:
            # 执行逆运动学求解
            # 使用数值方法求解逆运动学
            q_initial = np.zeros(self.robot.n)  # 初始关节角度
            
            # 使用 ikine_LM 方法 (Levenberg-Marquardt)
            sol = self.robot.ikine_LM(T_target, q0=q_initial)
            
            if sol.success:
                q_solution = sol.q
                print(f"逆运动学求解成功!")
                print(f"求解的关节角度: {np.degrees(q_solution)} (度)")
                
                # 验证解的正确性
                T_verify = self.robot.fkine(q_solution)
                position_error = np.linalg.norm(T_verify.t - target_position)
                print(f"位置误差: {position_error:.6f} m")
                
                # 检查关节限制
                self.check_joint_limits(q_solution)
                
                return q_solution
            else:
                print("逆运动学求解失败!")
                print(f"原因: {sol.reason}")
                return None
                
        except Exception as e:
            print(f"逆运动学计算出错: {e}")
            
            # 尝试其他方法
            try:
                print("尝试使用数值微分方法...")
                sol = self.robot.ikine_NR(T_target, q0=q_initial)
                if sol.success:
                    q_solution = sol.q
                    print(f"Newton-Raphson方法求解成功!")
                    print(f"求解的关节角度: {np.degrees(q_solution)} (度)")
                    return q_solution
                else:
                    print("Newton-Raphson方法也失败了")
            except:
                print("所有逆运动学方法都失败了")
                
            return None
            
    def check_joint_limits(self, q):
        """检查关节限制"""
        print("\n--- 关节限制检查 ---")
        for i, (angle, link) in enumerate(zip(q, self.robot.links)):
            if hasattr(link, 'qlim') and link.qlim is not None:
                qmin, qmax = link.qlim
                if qmin <= angle <= qmax:
                    status = "✓"
                else:
                    status = "✗"
                print(f"关节 {i+1}: {np.degrees(angle):.2f}° "
                      f"[{np.degrees(qmin):.1f}°, {np.degrees(qmax):.1f}°] {status}")
            else:
                print(f"关节 {i+1}: {np.degrees(angle):.2f}° [无限制]")
                
    def workspace_analysis(self):
        """工作空间分析"""
        print("\n=== 工作空间分析 ===")
        
        # 生成随机关节配置
        n_samples = 1000
        workspace_points = []
        
        print(f"正在计算 {n_samples} 个随机配置的工作空间...")
        
        for i in range(n_samples):
            # 在关节限制内生成随机关节角度
            q_random = []
            for link in self.robot.links:
                if hasattr(link, 'qlim') and link.qlim is not None:
                    qmin, qmax = link.qlim
                    q_random.append(np.random.uniform(qmin, qmax))
                else:
                    q_random.append(np.random.uniform(-np.pi, np.pi))
            
            q_random = np.array(q_random)
            
            # 计算末端执行器位置
            T = self.robot.fkine(q_random)
            workspace_points.append(T.t)
            
        workspace_points = np.array(workspace_points)
        
        # 分析工作空间
        print(f"工作空间统计:")
        print(f"X范围: [{workspace_points[:, 0].min():.3f}, {workspace_points[:, 0].max():.3f}] m")
        print(f"Y范围: [{workspace_points[:, 1].min():.3f}, {workspace_points[:, 1].max():.3f}] m")
        print(f"Z范围: [{workspace_points[:, 2].min():.3f}, {workspace_points[:, 2].max():.3f}] m")
        
        # 计算工作空间体积的近似值
        x_range = workspace_points[:, 0].max() - workspace_points[:, 0].min()
        y_range = workspace_points[:, 1].max() - workspace_points[:, 1].min()
        z_range = workspace_points[:, 2].max() - workspace_points[:, 2].min()
        approx_volume = x_range * y_range * z_range
        print(f"近似工作空间体积: {approx_volume:.3f} m³")
        
        return workspace_points
        
    def trajectory_planning_demo(self):
        """轨迹规划演示"""
        print("\n=== 轨迹规划演示 ===")
        
        # 定义起始和结束位置
        start_pos = [0.2, -0.15, 0.3]
        end_pos = [0.35, -0.25, 0.45]
        
        start_T = SE3.Rt(np.eye(3), start_pos)
        end_T = SE3.Rt(np.eye(3), end_pos)
        
        print(f"起始位置: {start_pos}")
        print(f"结束位置: {end_pos}")
        
        try:
            # 求解起始和结束的关节角度
            q_start = self.robot.ikine_LM(start_T).q
            q_end = self.robot.ikine_LM(end_T).q
            
            if q_start is None or q_end is None:
                print("无法求解起始或结束位置的逆运动学")
                return
                
            print(f"起始关节角度: {np.degrees(q_start)} (度)")
            print(f"结束关节角度: {np.degrees(q_end)} (度)")
            
            # 生成关节空间轨迹
            steps = 50
            t = np.linspace(0, 1, steps)
            
            # 简单线性插值
            q_trajectory = []
            for i in range(steps):
                q_interp = q_start + t[i] * (q_end - q_start)
                q_trajectory.append(q_interp)
                
            q_trajectory = np.array(q_trajectory)
            
            # 计算轨迹的笛卡尔空间位置
            cartesian_trajectory = []
            for q in q_trajectory:
                T = self.robot.fkine(q)
                cartesian_trajectory.append(T.t)
                
            cartesian_trajectory = np.array(cartesian_trajectory)
            
            print(f"生成了 {steps} 步的轨迹")
            print(f"轨迹长度: {np.sum(np.linalg.norm(np.diff(cartesian_trajectory, axis=0), axis=1)):.3f} m")
            
            return q_trajectory, cartesian_trajectory
            
        except Exception as e:
            print(f"轨迹规划出错: {e}")
            return None, None


def main():
    """主函数"""
    print("=== H1_2 机器人右臂运动学分析程序 ===")
    print("使用 Robotics Toolbox Python 版本")
    print("-" * 50)
    
    try:
        # 创建控制器实例
        controller = H1_2_RightArmController()
        
        # 初始化机器人
        controller.initialize_robot()
        
        # 显示机器人信息
        print(f"\n机器人信息:")
        print(f"名称: {controller.robot.name}")
        print(f"自由度: {controller.robot.n}")
        print(f"连杆数: {len(controller.robot.links)}")
        
        # 演示正运动学
        controller.demonstrate_forward_kinematics()
        
        # 演示逆运动学
        q_solution = controller.demonstrate_inverse_kinematics()
        
        # 工作空间分析
        workspace_points = controller.workspace_analysis()
        
        # 轨迹规划演示
        q_traj, cart_traj = controller.trajectory_planning_demo()
        
        print("\n=== 程序执行完成 ===")
        print("如需可视化结果，请确保安装了matplotlib并运行相应的绘图代码")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()