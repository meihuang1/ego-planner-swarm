#!/usr/bin/env python3
"""
将 nav_msgs/Odometry 转为 NavSatFix + Imu
  * ENU→WGS84 使用 pyproj
  * 可选白噪声
  * 支持多架无人机（launch 传 drone_id）
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

# ---------------- 可调参数（也可写在 ROS param） ----------------
GPS_NOISE_STD = 0.25            # [m]   GPS位置1σ噪声
GPS_VEL_STD = 0.25              # [m]   GPS速度1σ噪声

ACC_LP_ALPHA  = 0.8             # 一阶低通滤波系数 (0~1)，越大越平滑
# ---------------- normal noise ----------------
IMU_GYR_STD   = 0.01            # [rad/s] 角速度1σ噪声
IMU_ACC_STD   = 0.05            # [m/s²] 线加速度1σ噪声
ORI_NOISE_STD_DEG = 0.2         # 姿态噪声1σ，单位度

# ---------------- 0 noise ----------------
# IMU_GYR_STD   = 0            # [rad/s] 角速度1σ噪声
# IMU_ACC_STD   = 0            # [m/s²] 线加速度1σ噪声
# ORI_NOISE_STD_DEG = 0         # 姿态噪声1σ，单位度


ORI_NOISE_STD     = np.deg2rad(ORI_NOISE_STD_DEG)  # 转成弧度


MAG_NOISE_STD = 5e-6

# ---------------- 0 噪声 ----------------
# GPS_NOISE_STD = 0
# IMU_GYR_STD   = 0         # [rad/s]
# IMU_ACC_STD   = 0         # [m/s^2]
# ACC_LP_ALPHA  = 0          # 一阶低通 (0~1)，越大越平滑

# # 旋转的噪声
# ORI_NOISE_STD_DEG = 0       # 1σ = 0.2°

gps_var = GPS_NOISE_STD**2
ori_var = ORI_NOISE_STD**2
imu_gyr_var = IMU_GYR_STD**2
imu_acc_var = IMU_ACC_STD**2


# --------------------------------------------------------------

class Odom2GpsImu:
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        # ----------- 地理原点 (ENU 0,0,0) ----------
        self.lat0 = rospy.get_param('~origin_lat', 30.0)
        self.lon0 = rospy.get_param('~origin_lon', 120.0)
        self.alt0 = rospy.get_param('~origin_alt', 0.0)
        
        self.init_x = rospy.get_param('~init_x', 0.0)
        self.init_y = rospy.get_param('~init_y', 0.0)
        self.init_z = rospy.get_param('~init_z', 0.0)
        
        # pyproj 变换：WGS84 ↔ ENU( East, North, Up )

        enu_crs = (
            f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.lon0} "
            "+k=1 +x_0=0 +y_0=0 +ellps=WGS84"
        )
        self.trans_lla2enu = Transformer.from_crs("epsg:4326", enu_crs, always_xy=True)
        self.trans_enu2lla = Transformer.from_crs(enu_crs, "epsg:4326", always_xy=True)

        # ------- 订阅/发布 ---------
        odom_topic = f"/drone_{drone_id}_visual_slam/odom"
        rospy.Subscriber(odom_topic, Odometry, self.cb_odom, queue_size=50)

        self.pub_gps = rospy.Publisher(f"/drone_{drone_id}/gps",
                                       NavSatFix, queue_size=50)
        self.pub_imu = rospy.Publisher(f"/drone_{drone_id}/imu",
                                       Imu, queue_size=50)

        self.pub_mag = rospy.Publisher(f"/drone_{drone_id}/mag", MagneticField, queue_size=50)
        self.pub_gps_vel = rospy.Publisher(f"/drone_{drone_id}/gps_vel", TwistStamped, queue_size=50)

        # --- 差分所需缓存 ---
        self.last_vel   = None
        self.last_stamp = None
        self.lp_acc     = np.zeros(3)  # 低通滤波后的线加速度

        # --- 保持兼容性 ---
        self.euler = np.zeros(3)  # 初始化欧拉角（roll, pitch, yaw）- 保持原有逻辑
        
        # --- 改进的状态追踪 ---
        self.integrated_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # 积分四元数 (x,y,z,w)
        self.bias_gyro = np.zeros(3)  # 陀螺仪bias估计
        self.drift_correction_count = 0  # 漂移修正计数器
        self.last_correction_time = rospy.Time.now()
        
        # 漂移检测参数
        self.max_drift_threshold = 5.0  # 最大允许漂移距离(m)
        self.correction_interval = 2.0  # 漂移修正间隔(s)
        rospy.loginfo(f"[odom_to_gps_imu] Started for drone {drone_id}")

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        self.static_sent = False


    def cb_odom(self, odom):          # type: (Odom2GpsImu, Odometry) -> None
        """将 /odom 转成 NavSatFix + Imu 并发布"""
        stamp = odom.header.stamp
        
        # ========== 0. 设置 Frame ==========
        frame_prefix = f"drone_{self.drone_id}"

        # ========== 1. 取 ENU 坐标 ==========
        x_e = odom.pose.pose.position.x + self.init_x
        y_n = odom.pose.pose.position.y + self.init_y
        z_u = odom.pose.pose.position.z + self.init_z

        # ===== 2. ENU(x,y) → lon/lat，Z 直接平移 =====
        lon, lat = self.trans_enu2lla.transform(x_e, y_n)
        alt      = z_u + self.alt0                    # 不做高度投影

        # ---- GPS 白噪声 ----
        gps_noise = np.random.randn(3) * GPS_NOISE_STD
        lat += gps_noise[1] / 111320.0
        lon += gps_noise[0] / (40075000.0 * np.cos(radians(lat)) / 360.0)
        alt += gps_noise[2]

        gps_msg                 = NavSatFix()
        gps_msg.header          = odom.header
        gps_msg.header.frame_id = f"{frame_prefix}/gps_link"
        gps_msg.latitude        = lat
        gps_msg.longitude       = lon
        gps_msg.altitude        = alt
        gps_msg.status.status   = 0         # STATUS_FIX
        gps_msg.status.service  = 1         # SERVICE_GPS

        gps_msg.position_covariance = [gps_var, 0, 0,
                                    0, gps_var, 0,
                                    0, 0, gps_var]

        gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN  # = 2

        self.pub_gps.publish(gps_msg)

        # ========== 3. 陀螺仪：角速度 + 白噪声 ==========
        omega = np.array([odom.twist.twist.angular.x,
                        odom.twist.twist.angular.y,
                        odom.twist.twist.angular.z]) \
                + np.random.randn(3) * IMU_GYR_STD
        # ① 原始四元数
        q_raw = odom.pose.pose.orientation
        q_raw_np = np.array([q_raw.x, q_raw.y, q_raw.z, q_raw.w])

        # ===== 4. 角速度积分 ——欧拉角 → 四元数 =====
        if self.last_stamp is not None:
            dt = (stamp - self.last_stamp).to_sec()
        else:
            dt = 0.01                                 # 首帧假设 100 Hz
        self.euler += omega * dt                      # 欧拉角积分
        
        q_real = np.array([q_raw.x, q_raw.y, q_raw.z, q_raw.w])
        # ========== 5. 线加速度估算（世界系） ==========
        vel_now = np.array([odom.twist.twist.linear.x,
                            odom.twist.twist.linear.y,
                            odom.twist.twist.linear.z])

        if self.last_vel is not None and dt > 1e-4:
            acc_est = (vel_now - self.last_vel) / dt
        else:
            acc_est = np.zeros(3)

        # ======== 补偿重力：ENU Z 向上 =========
        acc_est[2] += 9.80665

        # ========== 加速度从世界坐标系旋转到机体坐标系 ==========
        # 四元数 q_dyn 是 world -> body
        q_world2body = quaternion_inverse(q_real)              # 形式: (x, y, z, w)
        acc_world_q = np.append(acc_est, 0.0)                 # 扩展为四元数 (x, y, z, 0)

        tmp = quaternion_multiply(q_world2body, acc_world_q)
        acc_body_q = quaternion_multiply(tmp, q_real)
        acc_body = acc_body_q[:3]                             # 机体坐标系下的加速度

        # ========== 滤波 + 噪声，在机体坐标系下进行 ==========
        self.lp_acc = ACC_LP_ALPHA * self.lp_acc + (1 - ACC_LP_ALPHA) * acc_body
        acc_noisy   = self.lp_acc + np.random.randn(3) * IMU_ACC_STD

        
        # ========== 6. 组装并发布 IMU ==========
        imu_msg                       = Imu()
        imu_msg.header                = odom.header
        imu_msg.header.frame_id       = f"{frame_prefix}/imu_link"

        # ========== 给旋转加一点噪声 ==========

        # ② 随机小角度（roll, pitch, yaw）
        dtheta = np.random.randn(3) * ORI_NOISE_STD   # rad

        # ③ 误差四元数  q_err = [dq_x dq_y dq_z dq_w]
        q_err = quaternion_from_euler(*dtheta)        # tf 返回 (x,y,z,w)

        # ④ 乘到原四元数 → 得到带噪声的新姿态
        q_noisy = quaternion_multiply(q_raw_np, q_err)  # 顺序: 先 q_raw 再 q_err

        # ⑤ 写回消息
        imu_msg.orientation.x, imu_msg.orientation.y, \
        imu_msg.orientation.z, imu_msg.orientation.w = q_noisy
                
        imu_msg.orientation_covariance         = [ori_var]*3 + [0]*6
        imu_msg.orientation_covariance[4]      = ori_var
        imu_msg.orientation_covariance[8]      = ori_var
        imu_msg.angular_velocity_covariance    = [imu_gyr_var]*9
        imu_msg.linear_acceleration_covariance = [imu_acc_var]*9
        
        # 生成 角速度 和加速度
        imu_msg.angular_velocity      = Vector3(*omega)
        imu_msg.linear_acceleration   = Vector3(*acc_noisy)
        self.pub_imu.publish(imu_msg)
        
        # ========== 7. 模拟地磁：world磁场 → body ==========
        # 地磁方向：世界系 ENU 表示下（默认沿 x 方向）
        mag_enu = np.array([1.0, 0.0, 0.0])   # 可换成 [0.707, 0.707, 0.0] 代表东北方向

        # 四元数 q_world2body = q_real⁻¹
        mag_body = quaternion_multiply(
            quaternion_multiply(quaternion_inverse(q_real), np.append(mag_enu, 0.0)),
            q_real
        )[:3]

        # 加一点噪声
        mag_noisy = mag_body + np.random.randn(3) * MAG_NOISE_STD

        # 生成磁力计消息（可选是否发布为 sensor_msgs/MagneticField）
        mag_msg = MagneticField()
        mag_msg.header = imu_msg.header
        mag_msg.magnetic_field = Vector3(*mag_noisy)
        mag_msg.magnetic_field_covariance = [MAG_NOISE_STD ** 2]*9
        self.pub_mag.publish(mag_msg)
        
        gps_vel_noise = np.random.randn(3) * GPS_VEL_STD
        gps_vel = vel_now + gps_vel_noise

        gps_vel_msg = TwistStamped()
        gps_vel_msg.header = imu_msg.header
        gps_vel_msg.twist.linear = Vector3(*gps_vel)
        self.pub_gps_vel.publish(gps_vel_msg)

        
        
if __name__ == "__main__":
    rospy.init_node("odom_to_gps_imu")
    drone_id = rospy.get_param("~drone_id", 0)
    Odom2GpsImu(int(drone_id))
    rospy.spin()
