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
            print(pos_est)
    last_t = t

if __name__ == "__main__":
    rospy.init_node("imu_to_odom")
    rospy.Subscriber("/drone_0/imu", Imu, imu_callback)
    odom_pub = rospy.Publisher("integrated_odom", Odometry, queue_size=10)
    rospy.spin()
