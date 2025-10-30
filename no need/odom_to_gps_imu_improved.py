#!/usr/bin/env python3
"""
改进版IMU生成器 - 减少积分漂移
主要改进：
1. 四元数积分替代欧拉角积分
2. 陀螺仪bias估计和补偿
3. 定期漂移修正
4. 更精确的加速度计算
"""
import rospy, numpy as np, tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Imu, MagneticField
from geometry_msgs.msg import Vector3, TwistStamped
from std_msgs.msg import Header
from pyproj import Transformer
from math import radians
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply, quaternion_inverse
import tf2_ros
from geometry_msgs.msg import TransformStamped

# ---------------- 可调参数 ----------------
GPS_NOISE_STD = 0.25            # [m]   GPS位置1σ噪声
GPS_VEL_STD = 0.25              # [m]   GPS速度1σ噪声

ACC_LP_ALPHA  = 0.8             # 一阶低通滤波系数 (0~1)，越大越平滑
IMU_GYR_STD   = 0.005           # [rad/s] 角速度1σ噪声 (降低噪声)
IMU_ACC_STD   = 0.03            # [m/s²] 线加速度1σ噪声 (降低噪声)
# IMU_GYR_STD = 0
# IMU_ACC_STD = 0

# GPS_NOISE_STD = 0            # [m]   GPS位置1σ噪声
# GPS_VEL_STD = 0             # [m]   GPS速度1σ噪声

ORI_NOISE_STD_DEG = 0.1         # 姿态噪声1σ，单位度 (降低噪声)

ORI_NOISE_STD = np.deg2rad(ORI_NOISE_STD_DEG)
MAG_NOISE_STD = 5e-6

gps_var = GPS_NOISE_STD**2
ori_var = ORI_NOISE_STD**2
imu_gyr_var = IMU_GYR_STD**2
imu_acc_var = IMU_ACC_STD**2

class ImprovedOdom2GpsImu:
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        
        # ----------- 地理原点 ----------
        self.lat0 = rospy.get_param('~origin_lat', 30.0)
        self.lon0 = rospy.get_param('~origin_lon', 120.0)
        self.alt0 = rospy.get_param('~origin_alt', 0.0)
        
        self.init_x = rospy.get_param('~init_x', 0.0)
        self.init_y = rospy.get_param('~init_y', 0.0)
        self.init_z = rospy.get_param('~init_z', 0.0)
        
        # pyproj 变换
        enu_crs = (
            f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.lon0} "
            "+k=1 +x_0=0 +y_0=0 +ellps=WGS84"
        )
        self.trans_lla2enu = Transformer.from_crs("epsg:4326", enu_crs, always_xy=True)
        self.trans_enu2lla = Transformer.from_crs(enu_crs, "epsg:4326", always_xy=True)

        # 订阅/发布
        odom_topic = f"/drone_{drone_id}_visual_slam/odom"
        rospy.Subscriber(odom_topic, Odometry, self.cb_odom, queue_size=50)

        self.pub_gps = rospy.Publisher(f"/drone_{drone_id}/gps", NavSatFix, queue_size=50)
        self.pub_imu = rospy.Publisher(f"/drone_{drone_id}/imu", Imu, queue_size=50)
        self.pub_imu_pure = rospy.Publisher(f"/drone_{drone_id}/imu_pure", Imu, queue_size=50)
        self.pub_mag = rospy.Publisher(f"/drone_{drone_id}/mag", MagneticField, queue_size=50)
        self.pub_gps_vel = rospy.Publisher(f"/drone_{drone_id}/gps_vel", TwistStamped, queue_size=50)

        # 状态变量
        self.last_vel = None
        self.last_pos = None
        self.last_stamp = None
        self.lp_acc = np.zeros(3)
        
        # 改进的IMU状态管理
        self.integrated_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # (x,y,z,w)
        self.bias_gyro = np.zeros(3)  # 陀螺仪bias估计
        self.bias_acc = np.zeros(3)   # 加速度计bias估计
        
        # 漂移修正参数
        self.drift_correction_interval = 3.0  # 每3秒修正一次
        self.last_correction_time = rospy.Time.now()
        self.correction_strength = 0.2  # 修正强度
        
        # 初始化完成标志
        self.initialized = False
        
        rospy.loginfo(f"[ImprovedOdom2GpsImu] Started for drone {drone_id}")

    def normalize_quaternion(self, q):
        """归一化四元数"""
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / norm

    def quaternion_multiply_np(self, q1, q2):
        """四元数乘法 (x,y,z,w)格式"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def quaternion_conjugate(self, q):
        """四元数共轭"""
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def integrate_quaternion_rk4(self, q, omega, dt):
        """使用RK4方法积分四元数"""
        def quaternion_derivative(q, w):
            # q_dot = 0.5 * q * [0, wx, wy, wz]
            w_q = np.array([w[0], w[1], w[2], 0.0])
            return 0.5 * self.quaternion_multiply_np(q, w_q)
        
        if dt <= 0:
            return q
            
        k1 = quaternion_derivative(q, omega)
        k2 = quaternion_derivative(q + 0.5*dt*k1, omega)
        k3 = quaternion_derivative(q + 0.5*dt*k2, omega)
        k4 = quaternion_derivative(q + dt*k3, omega)
        
        q_new = q + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.normalize_quaternion(q_new)

    def quaternion_error(self, q1, q2):
        """计算两个四元数的误差"""
        q1_conj = self.quaternion_conjugate(q1)
        return self.quaternion_multiply_np(q1_conj, q2)

    def slerp(self, q1, q2, t):
        """球面线性插值"""
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:
            # 线性插值
            result = q1 + t * (q2 - q1)
            return self.normalize_quaternion(result)
        
        theta_0 = np.arccos(abs(dot))
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2

    def estimate_acceleration_improved(self, vel_now, pos_now, dt):
        """改进的加速度估计"""
        if self.last_vel is None or self.last_pos is None or dt <= 1e-6:
            return np.zeros(3)
        
        # 方法1: 速度差分
        acc_vel = (vel_now - self.last_vel) / dt
        
        # 方法2: 位置二阶差分（更平滑，但有延迟）
        if hasattr(self, 'last_last_pos') and hasattr(self, 'last_dt'):
            if self.last_dt > 1e-6:
                acc_pos = (pos_now - 2*self.last_pos + self.last_last_pos) / (dt * self.last_dt)
            else:
                acc_pos = acc_vel
        else:
            acc_pos = acc_vel
        
        # 加权融合两种方法
        alpha = 0.7  # 更信任速度差分
        acc_est = alpha * acc_vel + (1-alpha) * acc_pos
        
        # 存储历史信息
        if hasattr(self, 'last_pos'):
            self.last_last_pos = self.last_pos.copy()
        if hasattr(self, 'last_dt'):
            self.last_dt = dt
        else:
            self.last_dt = dt
            
        return acc_est

    def cb_odom(self, odom):
        """改进的里程计处理回调"""
        stamp = odom.header.stamp
        frame_prefix = f"drone_{self.drone_id}"

        # ========== 位置和GPS处理 ==========
        x_e = odom.pose.pose.position.x + self.init_x
        y_n = odom.pose.pose.position.y + self.init_y
        z_u = odom.pose.pose.position.z + self.init_z
        pos_now = np.array([x_e, y_n, z_u])

        lon, lat = self.trans_enu2lla.transform(x_e, y_n)
        alt = z_u + self.alt0

        # GPS噪声
        gps_noise = np.random.randn(3) * GPS_NOISE_STD
        lat += gps_noise[1] / 111320.0
        lon += gps_noise[0] / (40075000.0 * np.cos(radians(lat)) / 360.0)
        alt += gps_noise[2]

        # 发布GPS
        gps_msg = NavSatFix()
        gps_msg.header = odom.header
        gps_msg.header.frame_id = f"{frame_prefix}/gps_link"
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        gps_msg.altitude = alt
        gps_msg.status.status = 0
        gps_msg.status.service = 1
        gps_msg.position_covariance = [gps_var, 0, 0, 0, gps_var, 0, 0, 0, gps_var]
        gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.pub_gps.publish(gps_msg)

        # ========== 改进的IMU处理 ==========
        q_raw = odom.pose.pose.orientation
        q_raw_np = np.array([q_raw.x, q_raw.y, q_raw.z, q_raw.w])
        
        omega_true = np.array([odom.twist.twist.angular.x,
                              odom.twist.twist.angular.y,
                              odom.twist.twist.angular.z])
        
        vel_now = np.array([odom.twist.twist.linear.x,
                           odom.twist.twist.linear.y,
                           odom.twist.twist.linear.z])

        # 时间步长
        if self.last_stamp is not None:
            dt = (stamp - self.last_stamp).to_sec()
        else:
            dt = 0.01
            self.integrated_quaternion = q_raw_np.copy()  # 初始化
            self.initialized = True

        # ========== 漂移修正机制 ==========
        current_time = stamp
        time_since_correction = (current_time - self.last_correction_time).to_sec()
        
        if self.initialized and time_since_correction > self.drift_correction_interval:
            # 计算积分四元数与真实四元数的误差
            q_error = self.quaternion_error(self.integrated_quaternion, q_raw_np)
            error_angle = 2 * np.arccos(np.clip(abs(q_error[3]), 0, 1))
            
            if error_angle > 0.05:  # 误差超过3度才修正
                # 估计陀螺仪bias
                if dt > 1e-6:
                    error_axis = q_error[:3] / (np.sin(error_angle/2) + 1e-6)
                    bias_correction = 0.05 * error_axis / dt  # 保守的bias修正
                    self.bias_gyro += bias_correction
                    
                    # 限制bias大小
                    max_bias = 0.1  # rad/s
                    self.bias_gyro = np.clip(self.bias_gyro, -max_bias, max_bias)
                
                # 软修正积分四元数
                self.integrated_quaternion = self.slerp(
                    self.integrated_quaternion, q_raw_np, self.correction_strength)
                self.integrated_quaternion = self.normalize_quaternion(self.integrated_quaternion)
                
                rospy.loginfo_throttle(10.0, 
                    f"IMU drift correction: error={np.degrees(error_angle):.1f}°, "
                    f"bias={np.linalg.norm(self.bias_gyro):.4f}")
                
            self.last_correction_time = current_time

        # ========== 角速度处理 ==========
        # 应用bias补偿
        omega_corrected = omega_true - self.bias_gyro
        
        # 积分四元数
        if self.initialized:
            self.integrated_quaternion = self.integrate_quaternion_rk4(
                self.integrated_quaternion, omega_corrected, dt)

        # 添加测量噪声
        omega_noisy = omega_corrected + np.random.randn(3) * IMU_GYR_STD

        # ========== 加速度处理 ==========
        acc_est = self.estimate_acceleration_improved(vel_now, pos_now, dt)
        acc_est[2] += 9.80665  # 重力补偿

        # 转换到机体坐标系
        q_world2body = quaternion_inverse(q_raw_np)
        acc_world_q = np.append(acc_est, 0.0)
        tmp = quaternion_multiply(q_world2body, acc_world_q)
        acc_body_q = quaternion_multiply(tmp, q_raw_np)
        acc_body = acc_body_q[:3]

        # 加速度计bias补偿和滤波
        acc_body_corrected = acc_body - self.bias_acc
        self.lp_acc = ACC_LP_ALPHA * self.lp_acc + (1 - ACC_LP_ALPHA) * acc_body_corrected
        acc_noisy = self.lp_acc + np.random.randn(3) * IMU_ACC_STD

        # ========== 组装IMU消息 ==========
        imu_msg = Imu()
        imu_msg.header = odom.header
        imu_msg.header.frame_id = f"{frame_prefix}/imu_link"

        # 使用原始四元数（而非积分的），因为我们主要关注角速度和加速度的改进
        # 在实际应用中，姿态会由融合算法处理
        imu_msg.orientation.x, imu_msg.orientation.y, \
        imu_msg.orientation.z, imu_msg.orientation.w = q_raw_np

        imu_msg.orientation_covariance = [ori_var]*3 + [0]*6
        imu_msg.orientation_covariance[4] = ori_var
        imu_msg.orientation_covariance[8] = ori_var
        imu_msg.angular_velocity_covariance = [imu_gyr_var]*9
        imu_msg.linear_acceleration_covariance = [imu_acc_var]*9

        imu_msg.angular_velocity = Vector3(*omega_noisy)
        imu_msg.linear_acceleration = Vector3(*acc_noisy)
        self.pub_imu.publish(imu_msg)
        
        # publish pure imu - 使用无噪声的原始数据
        imu_msg_pure = Imu()
        imu_msg_pure.header = odom.header
        imu_msg_pure.header.frame_id = f"{frame_prefix}/imu_link"
        imu_msg_pure.orientation.x, imu_msg_pure.orientation.y, \
        imu_msg_pure.orientation.z, imu_msg_pure.orientation.w = q_raw_np
        
        # 使用无噪声的原始角速度和加速度
        imu_msg_pure.angular_velocity = Vector3(*omega_corrected)  # 无噪声，但有bias补偿
        imu_msg_pure.linear_acceleration = Vector3(*acc_body_corrected)  # 无噪声，但有bias补偿
        
        # 设置协方差矩阵 - 对于pure IMU，协方差应该很小
        imu_msg_pure.orientation_covariance = [0.001]*3 + [0]*6  # 很小的姿态噪声
        imu_msg_pure.orientation_covariance[4] = 0.001
        imu_msg_pure.orientation_covariance[8] = 0.001
        imu_msg_pure.angular_velocity_covariance = [0.0001]*9  # 很小的角速度噪声
        imu_msg_pure.linear_acceleration_covariance = [0.001]*9  # 很小的加速度噪声
        
        self.pub_imu_pure.publish(imu_msg_pure)

        # ========== 磁力计 ==========
        mag_enu = np.array([1.0, 0.0, 0.0])
        mag_body = quaternion_multiply(
            quaternion_multiply(quaternion_inverse(q_raw_np), np.append(mag_enu, 0.0)),
            q_raw_np
        )[:3]
        mag_noisy = mag_body + np.random.randn(3) * MAG_NOISE_STD

        mag_msg = MagneticField()
        mag_msg.header = imu_msg.header
        mag_msg.magnetic_field = Vector3(*mag_noisy)
        mag_msg.magnetic_field_covariance = [MAG_NOISE_STD ** 2]*9
        self.pub_mag.publish(mag_msg)

        # ========== GPS速度 ==========
        gps_vel_noise = np.random.randn(3) * GPS_VEL_STD
        gps_vel = vel_now + gps_vel_noise

        gps_vel_msg = TwistStamped()
        gps_vel_msg.header = imu_msg.header
        gps_vel_msg.twist.linear = Vector3(*gps_vel)
        self.pub_gps_vel.publish(gps_vel_msg)

        # 更新历史状态
        self.last_vel = vel_now.copy()
        self.last_pos = pos_now.copy()
        self.last_stamp = stamp

if __name__ == "__main__":
    rospy.init_node("improved_odom_to_gps_imu")
    drone_id = rospy.get_param("~drone_id", 0)
    ImprovedOdom2GpsImu(int(drone_id))
    rospy.spin()