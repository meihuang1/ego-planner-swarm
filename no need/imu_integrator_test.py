#! /usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist



from utils import GRAVITY_Z, integrate_lin_acc, integrate_vel, integrate_ang_acc, integrate_ang_vel


pos_est = np.array([-140,-140,6])
vel_est = np.zeros(3)
ang_vel_est = np.zeros(3)
ang_quat_est = np.array([1,0,0,0])  # [w,x,y,z]

prev_lin_acc = np.zeros(3)
prev_ang_acc = np.zeros(3)
last_t = None

# 添加速度检测相关变量
vel_sign_history = []  # 存储速度符号历史
max_same_sign_count = 10  # 最大同号循环数

def generate_zero_mean_integral_noise(length, amplitude=0.001):
    """
    方法1: 生成均值为0且积分为0的随机噪声
    通过确保序列的和为0来实现积分为0
    """
    # 生成随机数
    noise = np.random.normal(0, amplitude, length)
    
    # 调整使得和为0（积分为0）
    noise_sum = np.sum(noise)
    noise -= noise_sum / length
    
    return noise

def generate_high_pass_noise(length, amplitude=0.001, cutoff_freq=0.1):
    """
    方法2: 高通滤波噪声，自然满足积分为0
    通过高通滤波器去除低频成分，保留高频噪声
    """
    # 生成白噪声
    white_noise = np.random.normal(0, amplitude, length)
    
    # 简单的高通滤波（去除直流分量）
    # 使用一阶差分滤波器
    filtered_noise = np.diff(white_noise, prepend=white_noise[0])
    
    # 归一化到目标幅度
    filtered_noise = filtered_noise * amplitude / np.std(filtered_noise)
    
    return filtered_noise

def generate_oscillating_noise(length, amplitude=0.001, periods=5):
    """
    方法3: 振荡噪声，通过正负交替确保积分为0
    生成多个周期的正弦波叠加，确保积分为0
    """
    t = np.linspace(0, 2*np.pi*periods, length)
    
    # 生成多个不同频率的正弦波
    noise = np.zeros(length)
    for i in range(1, 6):  # 5个不同频率
        freq = i * periods / length
        phase = np.random.uniform(0, 2*np.pi)
        noise += amplitude * np.sin(2*np.pi*freq*t + phase) / i
    
    # 确保积分为0
    noise_sum = np.sum(noise)
    noise -= noise_sum / length
    
    return noise

def generate_balanced_noise(length, amplitude=0.001):
    """
    方法4: 平衡噪声，通过正负配对确保积分为0
    生成随机数，然后通过配对正负值来平衡
    """
    # 生成随机数
    noise = np.random.uniform(-amplitude, amplitude, length)
    
    # 如果长度为奇数，确保最后一个数为0
    if length % 2 == 1:
        noise[-1] = 0
        length = length - 1
    
    # 将随机数分成两半，一半取负值
    half_length = length // 2
    first_half = noise[:half_length]
    second_half = -noise[half_length:2*half_length]
    
    # 随机打乱顺序
    combined = np.concatenate([first_half, second_half])
    np.random.shuffle(combined)
    
    return combined

def check_and_reset_velocity(velocity, lin_acc):
    """
    检测速度是否在连续多个循环内都是同号，如果是则强制置0
    只在IMU加速度接近0时才起作用，有实际加速度时正常积分
    """
    global vel_sign_history
    
    # 获取当前速度的符号（-1, 0, 1）
    current_signs = np.sign(velocity)
    
    # 检查IMU加速度是否在接近0的范围内（比如小于0.1 m/s²）
    acc_threshold = 0.1
    is_acc_near_zero = all(abs(acc) < acc_threshold for acc in lin_acc)
    
    # 如果有实际加速度，清空历史记录，正常积分
    if not is_acc_near_zero:
        vel_sign_history.clear()
        return velocity
    
    # 添加到历史记录
    vel_sign_history.append(current_signs)
    
    # 保持历史记录长度
    if len(vel_sign_history) > max_same_sign_count:
        vel_sign_history.pop(0)
    
    # 如果历史记录不足，直接返回原速度
    if len(vel_sign_history) < max_same_sign_count:
        return velocity
    
    # 检查每个轴是否在连续循环内都是同号
    for axis in range(3):
        axis_signs = [signs[axis] for signs in vel_sign_history]
        
        # 如果某个轴在连续循环内都是同号（排除0），则强制置0
        if all(sign != 0 for sign in axis_signs) and len(set(axis_signs)) == 1:
            # print(f"Axis {axis} has same sign for {max_same_sign_count} cycles with near-zero acc, resetting to 0")
            velocity[axis] = 0.0
    
    return velocity

def imu_callback(msg):
    global pos_est, vel_est, ang_vel_est, ang_quat_est
    global prev_lin_acc, prev_ang_acc, last_t

    t = msg.header.stamp.to_sec()
    if last_t is not None:
        # dt = t - last_t
        dt = 0.01
        
        if dt > 0:
            # --- 获取线加速度并补偿重力 ---
            lin_acc = np.array([msg.linear_acceleration.x,
                                msg.linear_acceleration.y,
                                msg.linear_acceleration.z])
            lin_acc[2] -= GRAVITY_Z

            # --- 获取角加速度 ---
            ang_acc = np.array([msg.angular_velocity.x,
                                msg.angular_velocity.y,
                                msg.angular_velocity.z])

            # --- 线速度积分 ---
            vel_est = integrate_lin_acc(vel_est, lin_acc, prev_lin_acc, dt)
            print("vel_est: ", vel_est)
            
            # --- 检测并重置异常偏移的速度 ---
            vel_est = check_and_reset_velocity(vel_est, lin_acc)
            
            # --- 位置积分 ---
            pos_est = integrate_vel(pos_est, vel_est, dt)
            prev_lin_acc = lin_acc

            # --- 角速度积分 ---
            ang_vel_est = integrate_ang_acc(ang_vel_est, ang_acc, prev_ang_acc, dt)
            prev_ang_acc = ang_acc

            # --- 姿态积分 ---
            ang_quat_est = integrate_ang_vel(ang_quat_est, ang_vel_est, dt)

            # --- 发布 Odometry ---
            odom_msg = Odometry()
            odom_msg.header.stamp = msg.header.stamp
            odom_msg.header.frame_id = "world"
            odom_msg.pose.pose = Pose(
                Point(*pos_est),
                Quaternion(*ang_quat_est[1:], ang_quat_est[0])
            )
            odom_msg.twist.twist.linear.x = vel_est[0]
            odom_msg.twist.twist.linear.y = vel_est[1]
            odom_msg.twist.twist.linear.z = vel_est[2]
            odom_msg.twist.twist.angular.x = ang_vel_est[0]
            odom_msg.twist.twist.angular.y = ang_vel_est[1]
            odom_msg.twist.twist.angular.z = ang_vel_est[2]
            odom_pub.publish(odom_msg)
            print(f"Position: {pos_est}")
            # print(f"Velocity: {vel_est}")
            print("---")
    last_t = t

if __name__ == "__main__":
    rospy.init_node("imu_to_odom")
    rospy.Subscriber("/drone_0/imu", Imu, imu_callback)
    odom_pub = rospy.Publisher("integrated_odom", Odometry, queue_size=10)
    rospy.spin()
