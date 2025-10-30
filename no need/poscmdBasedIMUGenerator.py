#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# -------- 全局变量 --------
init_pos = None
init_quat = None
last_t = None

# 积分结果
pos_int = np.zeros(3)      # 位置积分结果
vel_int = np.zeros(3)      # 速度积分结果
orientation_int = np.zeros(3)  # 姿态积分结果

# 上一次的速度和角速度（用于计算加速度）
last_vel = None
last_ang_vel = None

# 噪声参数
noise_std_acc = 0.02       # m/s²
noise_std_ang_acc = 0.001  # rad/s²

# 重力向量
gravity = np.array([0, 0, -9.80665])

# -------- 线性加速度积分函数（带重力补偿） --------
def integrate_linear_acceleration(acc, dt):
    """
    积分线性加速度得到位置
    acc: 包含重力的加速度 (m/s²)
    dt: 时间间隔 (s)
    """
    global vel_int, pos_int
    
    # 重力补偿
    acc_corrected = acc - gravity
    
    # 积分加速度得到速度
    vel_int += acc_corrected * dt
    
    # 积分速度得到位置
    pos_int += vel_int * dt
    
    return pos_int

# -------- 角加速度积分函数 --------
def integrate_angular_acceleration(ang_acc, dt):
    """
    积分角加速度得到姿态
    ang_acc: 角加速度 (rad/s²)
    dt: 时间间隔 (s)
    """
    global orientation_int
    
    # 积分角加速度得到姿态（欧拉角）
    orientation_int += ang_acc * dt
    
    return orientation_int

# -------- 从位置命令计算角速度和角加速度 --------
def calculate_angular_quantities(pos_cmd, dt):
    """
    从位置命令计算角速度和角加速度
    pos_cmd: PositionCommand 消息
    dt: 时间间隔 (s)
    """
    # 从 yaw 和 yaw_dot 计算角速度
    # 假设 roll 和 pitch 为 0，只有 yaw 有变化
    roll = 0.0
    pitch = 0.0
    yaw = pos_cmd.yaw
    
    # 角速度：只有 Z 轴有 yaw_dot
    angular_velocity = np.array([0.0, 0.0, pos_cmd.yaw_dot])
    
    # 角加速度：从 yaw_dot 的差分计算
    # 这里需要存储上一次的 yaw_dot
    global last_yaw_dot
    if hasattr(calculate_angular_quantities, 'last_yaw_dot') and dt > 0:
        yaw_ddot = (pos_cmd.yaw_dot - calculate_angular_quantities.last_yaw_dot) / dt
    else:
        yaw_ddot = 0.0
    
    calculate_angular_quantities.last_yaw_dot = pos_cmd.yaw_dot
    
    angular_acceleration = np.array([0.0, 0.0, yaw_ddot])
    
    return angular_velocity, angular_acceleration

# -------- ROS 回调函数：处理里程计数据（只处理线性加速度） --------
def odom_callback(msg):
    global last_vel, last_t, init_pos
    
    t = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
    
    # 初始化
    if init_pos is None:
        init_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        rospy.loginfo(f"Initial position recorded: {init_pos}")
    
    # 获取当前速度
    current_vel = np.array([
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.linear.z
    ])
    
    # 计算加速度（从速度差分）
    if last_vel is not None and last_t is not None:
        dt = t - last_t
        if dt > 0:
            # 计算线加速度
            linear_acc = (current_vel - last_vel) / dt
            # 加上重力（模拟真实IMU数据）
            linear_acc += gravity
            # 添加噪声
            linear_acc_noisy = linear_acc + np.random.normal(0, noise_std_acc, size=3)
            
            # 调用积分函数
            pos_estimated = init_pos + integrate_linear_acceleration(linear_acc_noisy, dt)
            
            # 获取真实值用于对比
            pos_true = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])
            
            # 计算误差
            pos_error = pos_estimated - pos_true
            
            # 输出结果
            rospy.loginfo(
                f"\n=== ODOM CALLBACK ===\n"
                f"=== TIME: {t:.3f}s ===\n"
                f"=== INPUT DATA ===\n"
                f"Linear Acceleration: {linear_acc_noisy}\n"
                f"=== INTEGRATION RESULTS ===\n"
                f"Position: {pos_estimated}\n"
                f"=== GROUND TRUTH ===\n"
                f"Position: {pos_true}\n"
                f"=== ERROR ANALYSIS ===\n"
                f"Position Error: {pos_error}\n"
                f"=== INTEGRATION STATE ===\n"
                f"Integrated Velocity: {vel_int}\n"
                f"Integrated Position: {pos_int}\n"
            )
    
    # 保存当前状态
    last_vel = current_vel
    last_t = t

# -------- ROS 回调函数：处理位置命令数据（处理角速度和角加速度） --------
def pos_cmd_callback(msg):
    global last_t, init_quat
    
    t = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
    
    # 初始化
    if init_quat is None:
        # 从 yaw 构造初始四元数
        roll = 0.0
        pitch = 0.0
        yaw = msg.yaw
        init_quat = quaternion_from_euler(roll, pitch, yaw)
        rospy.loginfo(f"Initial quaternion recorded: {init_quat}")
    
    # 计算角速度和角加速度
    if last_t is not None:
        dt = t - last_t
        if dt > 0:
            angular_velocity, angular_acceleration = calculate_angular_quantities(msg, dt)
            
            # 添加噪声
            angular_acc_noisy = angular_acceleration + np.random.normal(0, noise_std_ang_acc, size=3)
            
            # 调用积分函数
            orientation_estimated = integrate_angular_acceleration(angular_acc_noisy, dt)
            
            # 获取真实值用于对比
            roll_true = 0.0
            pitch_true = 0.0
            yaw_true = msg.yaw
            euler_true = np.array([roll_true, pitch_true, yaw_true])
            
            # 计算误差
            orientation_error = orientation_estimated - euler_true
            
            # 输出结果
            rospy.loginfo(
                f"\n=== POS_CMD CALLBACK ===\n"
                f"=== TIME: {t:.3f}s ===\n"
                f"=== INPUT DATA ===\n"
                f"Yaw: {np.degrees(msg.yaw):.2f}°\n"
                f"Yaw_dot: {np.degrees(msg.yaw_dot):.2f}°/s\n"
                f"Angular Acceleration: {np.degrees(angular_acc_noisy):.2f}°/s²\n"
                f"=== INTEGRATION RESULTS ===\n"
                f"Orientation (euler): {np.degrees(orientation_estimated):.2f}°\n"
                f"=== GROUND TRUTH ===\n"
                f"Orientation (euler): {np.degrees(euler_true):.2f}°\n"
                f"=== ERROR ANALYSIS ===\n"
                f"Orientation Error: {np.degrees(orientation_error):.2f}°\n"
                f"=== INTEGRATION STATE ===\n"
                f"Integrated Orientation: {np.degrees(orientation_int):.2f}°\n"
            )
    
    last_t = t

if __name__ == '__main__':
    rospy.init_node('acceleration_integrator')
    last_t = None
    last_vel = None
    
    # 订阅两个话题
    rospy.Subscriber('/drone_0_visual_slam/odom', Odometry, odom_callback)
    rospy.Subscriber('/drone_0_planning/pos_cmd', PositionCommand, pos_cmd_callback)
    
    rospy.loginfo("Starting acceleration integrator...")
    rospy.loginfo("Subscribing to:")
    rospy.loginfo("  - /drone_0_visual_slam/odom (for linear acceleration)")
    rospy.loginfo("  - /drone_0_planning/pos_cmd (for angular quantities)")
    rospy.spin()