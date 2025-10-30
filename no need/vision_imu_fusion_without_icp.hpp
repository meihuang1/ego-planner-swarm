#ifndef VISION_IMU_FUSION_HPP
#define VISION_IMU_FUSION_HPP

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>

// 移除PCL相关头文件

#include <Eigen/Dense>
#include <co_navi_pkg/hpFnc.hpp>
struct State {
  Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d prev_lin_acc{Eigen::Vector3d::Zero()};
  Eigen::Vector3d noisy_vel{Eigen::Vector3d::Zero()};  // 添加噪声速度
  ros::Time last_t;
  ros::Time last_vel_t;
  bool has_prev{false};
  bool has_vel{false};
};

inline Eigen::Vector3d integrate_lin_acc(
    const Eigen::Vector3d& prev_vel,
    const Eigen::Vector3d& lin_acc,
    const Eigen::Vector3d& prev_lin_acc,
    double dt) {
  return prev_vel + 0.5 * (prev_lin_acc + lin_acc) * dt;
}

inline Eigen::Vector3d integrate_vel(
    const Eigen::Vector3d& prev_pos,
    const Eigen::Vector3d& vel,
    double dt) {
  return prev_pos + vel * dt;
}

class IMUVisionFusion {
public:
  IMUVisionFusion(ros::NodeHandle& nh) : nh_(nh) {
    nh_.param<int>("drone_id", drone_id_, 0);

    // 移除ICP参数

    // 初始位置参数
    nh_.param<float>("init_x", init_x_, 0.0f);
    nh_.param<float>("init_y", init_y_, 0.0f);
    nh_.param<float>("init_z", init_z_, 0.0f);

    // IMU参数
    nh_.param<bool>("compensate_gravity", compensate_gravity_, true);
    nh_.param<double>("gravity_z", gravity_z_, 9.80665);
    nh_.param<double>("fixed_dt", fixed_dt_, -1.0);  // <0 表示用时间戳差

    // 移除skip_icp参数
    nh_.param<std::string>("run_mode", mode_, "low");

    // 高精度模式选择
    nh_.param<bool>("using_high_precision", using_high_precision_, false);

    std::string topic_prefix = "/drone_" + std::to_string(drone_id_);

    // 移除地图初始化

    // 根据高精度模式选择话题名称
    std::string imu_topic, fused_topic, imu_odom_topic, vel_topic;
    if (using_high_precision_) {
      imu_topic = topic_prefix + "/imu_hp";
      fused_topic = topic_prefix + "/odom_fused_IV_hp";
      imu_odom_topic = topic_prefix + "/imu_integrator_res2";
      vel_topic = topic_prefix + "/vel_noisy_hp";
    } else {
      imu_topic = topic_prefix + "/imu";
      fused_topic = topic_prefix + "/odom_fused_IV";
      imu_odom_topic = topic_prefix + "/imu_integrator_res";
      vel_topic = topic_prefix + "/vel_noisy";
    }

    // 订阅器
    imu_sub_   = nh_.subscribe(imu_topic, 200, &IMUVisionFusion::imuCallback, this);
    vel_noise_sub_ = nh_.subscribe(vel_topic, 200, &IMUVisionFusion::velocityCallback, this);
    odom_vis_sub_ = nh.subscribe(topic_prefix + "_visual_slam/odom", 1, &IMUVisionFusion::odomVisCallback, this);
    // 发布器
    fused_pub_   = nh_.advertise<nav_msgs::Odometry>(fused_topic, 50);
    imu_pub_     = nh_.advertise<nav_msgs::Odometry>(imu_odom_topic, 50);

    // 初始化状态
    st_.pos = Eigen::Vector3d(init_x_, init_y_, init_z_);
    st_.vel.setZero();
    st_.prev_lin_acc.setZero();
    st_.has_prev = false;

    ROS_INFO("IMU fusion initialized for drone_%d (high_precision=%s)", 
             drone_id_, using_high_precision_ ? "true" : "false");
    ROS_INFO("Topics: IMU=%s, Fused=%s, IMU_odom=%s", 
             imu_topic.c_str(), fused_topic.c_str(), imu_odom_topic.c_str());
  }

private:
  // ===== ROS对象 =====
  ros::NodeHandle nh_;
  ros::Subscriber imu_sub_, odom_vis_sub_, vel_noise_sub_;
  ros::Publisher fused_pub_, imu_pub_;

  // ===== 参数 =====
  int drone_id_{0};
  float init_x_{0}, init_y_{0}, init_z_{0};
  bool compensate_gravity_{true};
  double gravity_z_{9.80665};
  double fixed_dt_{-1.0}; // 若>=0，则使用固定dt（秒）
  bool using_high_precision_{false}; // 高精度模式开关

  std::string mode_;
  // ===== 状态与缓存 =====
  State st_;

  // ===== 速度回调函数 =====
  void velocityCallback(const geometry_msgs::TwistStamped::ConstPtr& msg) {
    // 更新噪声速度
    st_.noisy_vel = Eigen::Vector3d(msg->twist.linear.x,
                                   msg->twist.linear.y,
                                   msg->twist.linear.z);
    st_.last_vel_t = msg->header.stamp;
    st_.has_vel = true;
  }

  // ===== IMU回调函数 =====
  void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    // 初始化
    if (!st_.has_prev) {
      st_.pos = Eigen::Vector3d(init_x_, init_y_, init_z_);
      st_.vel.setZero();
      st_.prev_lin_acc.setZero();
      st_.last_t = msg->header.stamp;
      st_.has_prev = true;
      publishIMUOdometry(msg->header.stamp);
      return;
    }

    // 使用固定时间间隔（与Python版本保持一致）
    double dt = 0.01;
    if (dt <= 0.0 || dt > 0.1) { // 简单防护：超大dt直接丢弃，避免数值爆炸
      st_.last_t = msg->header.stamp;
      return;
    }

    // 获取线性加速度并补偿重力（与Python版本逻辑一致）
    Eigen::Vector3d lin_acc(msg->linear_acceleration.x,
                            msg->linear_acceleration.y,
                            msg->linear_acceleration.z);
    if (compensate_gravity_) {
      lin_acc.z() -= gravity_z_;
    }

    // 线速度积分（与Python版本一致）
    st_.vel = integrate_lin_acc(st_.vel, lin_acc, st_.prev_lin_acc, dt);
    
    // 位置积分：如果有噪声速度数据，使用噪声速度；否则使用积分得到的速度
    if (st_.has_vel) {
      st_.pos = integrate_vel(st_.pos, st_.noisy_vel, dt);
    } else {
      st_.pos = integrate_vel(st_.pos, st_.vel, dt);
    }
    
    st_.prev_lin_acc = lin_acc;
    st_.last_t = msg->header.stamp;

    // 发布IMU积分结果
    publishIMUOdometry(msg->header.stamp);
    
    // 发布融合结果
    publishFusedOdometry(msg->header.stamp);
  }

  Eigen::Vector3d visPos_, visVel_;
  void odomVisCallback(const nav_msgs::Odometry::ConstPtr &msg)
  {
    Eigen::Vector3d pos(msg->pose.pose.position.x,
                        msg->pose.pose.position.y,
                        msg->pose.pose.position.z);
    Eigen::Vector3d vel(msg->twist.twist.linear.x,
                        msg->twist.twist.linear.y,
                        msg->twist.twist.linear.z);

    visPos_ = pos;
    visVel_ = vel;
  }

  // 移除所有ICP相关函数


  // ===== 发布函数 =====
  void publishIMUOdometry(const ros::Time& stamp) {
    nav_msgs::Odometry od;
    od.header.stamp = stamp;
    od.header.frame_id = "world";  // 与Python版本保持一致
    od.child_frame_id = "base_link";

    // 位置（来自IMU双积分）
    od.pose.pose.position.x = st_.pos.x();
    od.pose.pose.position.y = st_.pos.y();
    od.pose.pose.position.z = st_.pos.z();
    
    // 姿态（不处理旋转，使用单位四元数，与Python版本格式一致）
    od.pose.pose.orientation.w = 1.0;
    od.pose.pose.orientation.x = 0.0;
    od.pose.pose.orientation.y = 0.0;
    od.pose.pose.orientation.z = 0.0;

    // 线速度（来自IMU积分）
    od.twist.twist.linear.x = st_.vel.x();
    od.twist.twist.linear.y = st_.vel.y();
    od.twist.twist.linear.z = st_.vel.z();

    // 角速度（不处理旋转，设为0）
    od.twist.twist.angular.x = 0.0;
    od.twist.twist.angular.y = 0.0;
    od.twist.twist.angular.z = 0.0;

    imu_pub_.publish(od);
    
    // 打印位置信息（与Python版本保持一致）
    // ROS_INFO("Position: [%.2f, %.2f, %.2f]", st_.pos.x(), st_.pos.y(), st_.pos.z());
  }

  void publishFusedOdometry(const ros::Time& stamp) {
    nav_msgs::Odometry od;
    od.header.stamp = stamp;
    od.header.frame_id = "world";
    od.child_frame_id = "base_link";

    // 位置（来自IMU积分）
    od.pose.pose.position.x = st_.pos.x();
    od.pose.pose.position.y = st_.pos.y();
    od.pose.pose.position.z = st_.pos.z();
    
    // 姿态（不处理旋转，使用单位四元数）
    od.pose.pose.orientation.w = 1.0;
    od.pose.pose.orientation.x = 0.0;
    od.pose.pose.orientation.y = 0.0;
    od.pose.pose.orientation.z = 0.0;

    Eigen::Vector3d estVel = st_.vel;
    if(mode_ == "low"){
      Eigen::Vector3d outVel = outputProcessing(visVel_, st_.vel, mode_, true);
      estVel = outVel;
    }

    // 速度（来自IMU积分）
    od.twist.twist.linear.x = estVel.x();
    od.twist.twist.linear.y = estVel.y();
    od.twist.twist.linear.z = estVel.z();

    // 协方差
    double pvar = 0.05;
    od.pose.covariance[0] = od.pose.covariance[7] = od.pose.covariance[14] = pvar;

    fused_pub_.publish(od);
  }
};

#endif // VISION_IMU_FUSION_HPP