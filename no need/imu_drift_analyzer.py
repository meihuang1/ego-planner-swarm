#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from nav_msgs.msg import Odometry

class IMUDriftAnalyzer:
    def __init__(self):
        rospy.init_node('imu_drift_analyzer', anonymous=True)
        
        # 订阅话题
        self.pure_imu_sub = rospy.Subscriber('/pure_imu/odom', Odometry, self.pure_imu_callback)
        self.fusion_sub = rospy.Subscriber('/fusion/odom', Odometry, self.fusion_callback)
        self.ground_truth_sub = rospy.Subscriber('/drone_0/global_odom', Odometry, self.ground_truth_callback)
        
        # 统计信息
        self.start_time = None
        self.drift_stats = {
            'pure_imu_max_drift': 0.0,
            'fusion_max_drift': 0.0,
            'pure_imu_avg_drift': 0.0,
            'fusion_avg_drift': 0.0
        }
        
        # 定时器，每5秒输出一次统计信息
        self.timer = rospy.Timer(rospy.Duration(5.0), self.print_stats)
        
        rospy.loginfo("IMU漂移分析器已启动")
    
    def pure_imu_callback(self, msg):
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()
        
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.calculate_drift('pure_imu', pos)
    
    def fusion_callback(self, msg):
        if self.start_time is None:
            return
            
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.calculate_drift('fusion', pos)
    
    def ground_truth_callback(self, msg):
        self.ground_truth_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
    
    def calculate_drift(self, method, pos):
        if not hasattr(self, 'ground_truth_pos'):
            return
            
        drift = np.linalg.norm(np.array(pos) - np.array(self.ground_truth_pos))
        
        if method == 'pure_imu':
            self.drift_stats['pure_imu_max_drift'] = max(self.drift_stats['pure_imu_max_drift'], drift)
            self.drift_stats['pure_imu_avg_drift'] = drift  # 简化，只显示最新值
        elif method == 'fusion':
            self.drift_stats['fusion_max_drift'] = max(self.drift_stats['fusion_max_drift'], drift)
            self.drift_stats['fusion_avg_drift'] = drift  # 简化，只显示最新值
    
    def print_stats(self, event):
        rospy.loginfo("=" * 50)
        rospy.loginfo("IMU漂移分析报告")
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"纯IMU积分:")
        rospy.loginfo(f"  最大漂移: {self.drift_stats['pure_imu_max_drift']:.3f} m")
        rospy.loginfo(f"  当前漂移: {self.drift_stats['pure_imu_avg_drift']:.3f} m")
        rospy.loginfo(f"IMU+LiDAR融合:")
        rospy.loginfo(f"  最大漂移: {self.drift_stats['fusion_max_drift']:.3f} m")
        rospy.loginfo(f"  当前漂移: {self.drift_stats['fusion_avg_drift']:.3f} m")
        
        if self.drift_stats['pure_imu_avg_drift'] > 0 and self.drift_stats['fusion_avg_drift'] > 0:
            improvement = (self.drift_stats['pure_imu_avg_drift'] - self.drift_stats['fusion_avg_drift']) / self.drift_stats['pure_imu_avg_drift'] * 100
            rospy.loginfo(f"融合改善: {improvement:.1f}%")
        rospy.loginfo("=" * 50)

if __name__ == '__main__':
    try:
        analyzer = IMUDriftAnalyzer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 