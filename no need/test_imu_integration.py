#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU积分改进测试脚本
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

class IMUIntegrationTester:
    def __init__(self):
        rospy.init_node('imu_integration_tester')
        
        # 订阅器
        self.pure_imu_sub = rospy.Subscriber('/pure_imu/odom', Odometry, self.pure_imu_callback)
        self.fusion_sub = rospy.Subscriber('/fusion/odom', Odometry, self.fusion_callback)
        
        # 统计
        self.pure_imu_positions = []
        self.fusion_positions = []
        
        rospy.loginfo("IMU积分测试器已启动")
        
    def pure_imu_callback(self, msg):
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.pure_imu_positions.append(pos)
        
        if len(self.pure_imu_positions) % 100 == 0:
            self.analyze_performance()
    
    def fusion_callback(self, msg):
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.fusion_positions.append(pos)
    
    def analyze_performance(self):
        if len(self.pure_imu_positions) < 10:
            return
            
        # 计算漂移
        start_pos = np.array(self.pure_imu_positions[0])
        end_pos = np.array(self.pure_imu_positions[-1])
        drift = np.linalg.norm(end_pos - start_pos)
        
        rospy.loginfo("当前漂移: %.3f m", drift)

if __name__ == '__main__':
    try:
        tester = IMUIntegrationTester()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 