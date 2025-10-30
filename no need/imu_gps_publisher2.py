#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import TwistStamped, Vector3, Quaternion
from utils import quat_mult, quat_to_ang_vel, GRAVITY_Z
import tf.transformations as tf_trans
from pyproj import Transformer


prev_pos = None
prev_vel = np.zeros(3)
prev_quat = None
prev_ang_vel = np.zeros(3)
last_t = None
# IMU 噪声
ang_acc_noise_std = 0.001
lin_acc_noise_std = 0.01
ori_noise_std = 0.001

# GPS 噪声
GPS_NOISE_STD = 0.05
GPS_VEL_STD = 0.1
gps_var = GPS_NOISE_STD**2


class PublisherImuGps:
    def __init__(self):
        rospy.init_node("publisher_imu_gps2")

        # 参数（你可以改成从参数服务器加载）
        self.drone_id = rospy.get_param("~drone_id", 0)
        
        self.init_x = rospy.get_param("~init_x", 0.0)
        self.init_y = rospy.get_param("~init_y", 0.0)
        self.init_z = rospy.get_param("~init_z", 0.0)
        
        self.lat0 = rospy.get_param('~origin_lat', 30.0)
        self.lon0 = rospy.get_param('~origin_lon', 120.0)
        self.alt0 = rospy.get_param('~origin_alt', 0.0)
        
        self.prev_pos = None
        self.prev_vel = np.zeros(3)
        self.prev_quat = None
        self.prev_ang_vel = np.zeros(3)
        self.last_t = None


        # pyproj转换器：ENU <-> WGS84
        enu_crs = (
            f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.lon0} "
            "+k=1 +x_0=0 +y_0=0 +ellps=WGS84"
        )
        # 修复：正确的转换方向
        self.trans_lla2enu = Transformer.from_crs("epsg:4326", enu_crs, always_xy=True)
        self.trans_enu2lla = Transformer.from_crs(enu_crs, "epsg:4326", always_xy=True)

        # 发布器
        self.pub_imu = rospy.Publisher(f"/drone_{self.drone_id}/imu", Imu, queue_size=10)
        self.pub_gps = rospy.Publisher(f"/drone_{self.drone_id}/gps", NavSatFix, queue_size=10)
        self.pub_gps_vel = rospy.Publisher(f"/drone_{self.drone_id}/gps_vel", TwistStamped, queue_size=10)

        # 订阅视觉里程计
        rospy.Subscriber(f"/drone_{self.drone_id}_visual_slam/odom", Odometry, self.odom_callback)

        # 在 __init__ 里初始化积分变量
        self.int_pos_from_vel = np.array([-140.0,-140.0,6.0])   # 速度积分得到的位置
        self.int_pos_from_acc = np.array([-140.0,-140.0,6.0])   # 加速度积分得到的位置
        self.int_vel_from_acc = np.zeros(3)   # 加速度积分得到的速度

        self.acc_noise_vel = np.zeros(3)
        self.pos_noise_vel = np.zeros(3)
        
        rospy.loginfo("[imu_publisher_with_gps] Node started.")
        rospy.spin()

    def odom_callback(self, msg):
        # print(self.alt0, self.lat0, self.lon0)
        # print(f"/{self.frame_prefix}/imu")
        t = msg.header.stamp.to_sec()
        pos = np.array([msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z])
        quat = np.array([msg.pose.pose.orientation.w,
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z])
        
        if self.prev_pos is not None and self.prev_quat is not None and self.last_t is not None:
            # dt = t - self.last_t
            dt = 0.01
            if dt > 0:
                # --- 线加速度 ---
                vel = (pos - self.prev_pos) / dt
                lin_acc = (vel - self.prev_vel) / dt
                lin_acc[2] += GRAVITY_Z
                acc_noisy = np.random.normal(0, lin_acc_noise_std, 3)
                lin_acc_noisy = lin_acc + acc_noisy
                
                # --- 角加速度 ---
                ang_vel = quat_to_ang_vel(self.prev_quat, quat, dt)
                ang_acc = (ang_vel - self.prev_ang_vel) / dt
                ang_acc_noisy = ang_acc + np.random.normal(0, ang_acc_noise_std, 3)

                # --- 给 orientation 加噪声 ---
                # 1. 转成欧拉角
                euler = np.array(tf_trans.euler_from_quaternion(quat))
                # 2. 每个欧拉角加高斯噪声
                euler_noisy = euler + np.random.normal(0, ori_noise_std, 3)
                # 3. 转回四元数
                quat_noisy = tf_trans.quaternion_from_euler(*euler_noisy)

                # 速度一次积分 -> 位置
                self.int_pos_from_vel += vel * dt

                # 加速度两次积分 -> 位置
                lin_acc = lin_acc_noisy
                lin_acc[2] -= GRAVITY_Z
                self.int_vel_from_acc += lin_acc * dt
                self.int_pos_from_acc += self.int_vel_from_acc * dt

                self.acc_noise_vel += acc_noisy * dt
                self.pos_noise_vel += self.acc_noise_vel * dt
                
                # 打印结果
                print("真实 Odom 位置:", pos)
                print("速度积分位置:", self.int_pos_from_vel)
                print("加速度积分位置:", self.int_pos_from_acc)
                print("误差(vel):", np.linalg.norm(pos - self.int_pos_from_vel))
                print("误差(acc):", np.linalg.norm(pos - self.int_pos_from_acc))
                print("=" * 50)

                print("acc_noise_vel:", self.acc_noise_vel)
                print("pos_noise_vel:", self.pos_noise_vel)


                # --- 打包 IMU 消息 ---
                imu_msg = Imu()
                imu_msg.header.stamp = msg.header.stamp
                imu_msg.header.frame_id = "base_link"
                imu_msg.orientation.x = quat_noisy[1]
                imu_msg.orientation.y = quat_noisy[2]
                imu_msg.orientation.z = quat_noisy[3]
                imu_msg.orientation.w = quat_noisy[0]

                imu_msg.linear_acceleration.x = lin_acc_noisy[0]
                imu_msg.linear_acceleration.y = lin_acc_noisy[1]
                imu_msg.linear_acceleration.z = lin_acc_noisy[2]
                imu_msg.angular_velocity.x = ang_acc_noisy[0]
                imu_msg.angular_velocity.y = ang_acc_noisy[1]
                imu_msg.angular_velocity.z = ang_acc_noisy[2]

                # 协方差矩阵
                imu_msg.orientation_covariance = [ori_noise_std**2, 0, 0,
                                                0, ori_noise_std**2, 0,
                                                0, 0, ori_noise_std**2]
                imu_msg.angular_velocity_covariance = [ang_acc_noise_std**2, 0, 0,
                                                    0, ang_acc_noise_std**2, 0,
                                                    0, 0, ang_acc_noise_std**2]
                imu_msg.linear_acceleration_covariance = [lin_acc_noise_std**2, 0, 0,
                                                        0, lin_acc_noise_std**2, 0,
                                                        0, 0, lin_acc_noise_std**2]

                self.pub_imu.publish(imu_msg)

                self.prev_vel = vel
                self.prev_ang_vel = ang_vel

                # 发布 GPS 和 GPS 速度
                self.publish_gps_from_odom(msg)

        self.prev_pos = pos
        self.prev_quat = quat
        self.last_t = t

    def publish_gps_from_odom(self, odom):
        # ENU加初始化偏移
        x_e = odom.pose.pose.position.x + self.init_x
        y_n = odom.pose.pose.position.y + self.init_y
        z_u = odom.pose.pose.position.z + self.init_z

        # ENU -> 经纬度转换
        lon, lat = self.trans_enu2lla.transform(x_e, y_n)
        alt = z_u + self.alt0

        # GPS噪声（单位换算）
        gps_noise = np.random.randn(3) * GPS_NOISE_STD
        lat += gps_noise[1] / 111320.0
        lon += gps_noise[0] / (40075000.0 * np.cos(np.radians(lat)) / 360.0)
        alt += gps_noise[2]

        # 构造NavSatFix消息
        gps_msg = NavSatFix()
        gps_msg.header = odom.header
        gps_msg.header.frame_id = f"drone_{self.drone_id}/gps_link"
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        gps_msg.altitude = alt
        gps_msg.status.status = 0
        gps_msg.status.service = 1
        gps_msg.position_covariance = [gps_var, 0, 0,
                                       0, gps_var, 0,
                                       0, 0, gps_var]
        gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.pub_gps.publish(gps_msg)

        # GPS速度带噪声
        vel_now = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z])
        gps_vel_noise = np.random.randn(3) * GPS_VEL_STD
        gps_vel = vel_now + gps_vel_noise

        gps_vel_msg = TwistStamped()
        gps_vel_msg.header = odom.header
        gps_vel_msg.twist.linear.x = gps_vel[0]
        gps_vel_msg.twist.linear.y = gps_vel[1]
        gps_vel_msg.twist.linear.z = gps_vel[2]

        self.pub_gps_vel.publish(gps_vel_msg)


if __name__ == "__main__":
    try:
        PublisherImuGps()
    except rospy.ROSInterruptException:
        pass
