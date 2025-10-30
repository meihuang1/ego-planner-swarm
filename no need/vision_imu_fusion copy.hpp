#ifndef VISION_IMU_FUSION_HPP
#define VISION_IMU_FUSION_HPP

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp_nl.h>

#include <Eigen/Dense>
#include <co_navi_pkg/hpFnc.hpp>
struct State {
  Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d prev_lin_acc{Eigen::Vector3d::Zero()};
  ros::Time last_t;
  bool has_prev{false};
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

    // ICP参数
    nh_.param<float>("icp_max_distance", icp_max_distance_, 1.0);
    nh_.param<int>("icp_max_iter", icp_max_iter_, 40);
    nh_.param<float>("voxel_leaf_size", voxel_leaf_size_, 0.25f);
    nh_.param<int>("min_cloud_points", min_cloud_points_, 1000);

    // 初始位置参数
    nh_.param<float>("init_x", init_x_, 0.0f);
    nh_.param<float>("init_y", init_y_, 0.0f);
    nh_.param<float>("init_z", init_z_, 0.0f);

    // IMU参数
    nh_.param<bool>("compensate_gravity", compensate_gravity_, true);
    nh_.param<double>("gravity_z", gravity_z_, 9.80665);
    nh_.param<double>("fixed_dt", fixed_dt_, -1.0);  // <0 表示用时间戳差

    // 快速调试开关：设置为true时跳过ICP，只使用IMU积分
    nh_.param<bool>("skip_icp", skip_icp_, false);
    nh_.param<std::string>("run_mode", mode_, "low");

    // 高精度模式选择
    nh_.param<bool>("using_high_precision", using_high_precision_, false);

    std::string topic_prefix = "/drone_" + std::to_string(drone_id_);

    // 地图初始化
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    map_with_normals_.reset(new pcl::PointCloud<pcl::PointNormal>);
    map_ready_ = false;

    // 根据高精度模式选择话题名称
    std::string imu_topic, fused_topic, imu_odom_topic;
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
    map_sub_   = nh_.subscribe("/map_generator/global_cloud", 1, &IMUVisionFusion::mapCallback, this);
    imu_sub_   = nh_.subscribe(imu_topic, 200, &IMUVisionFusion::imuCallback, this);
    vel_noise_sub_ = nh_.subscribe(vel_topic, 200, &IMUVisionFusion::velocityCallback, this);
    cloud_sub_ = nh_.subscribe(topic_prefix + "_pcl_render_node/cloud", 5, &IMUVisionFusion::cloudCallback, this);
    odom_vis_sub_ = nh.subscribe(topic_prefix + "_visual_slam/odom", 1, &IMUVisionFusion::odomVisCallback, this);
    // 发布器
    fused_pub_   = nh_.advertise<nav_msgs::Odometry>(fused_topic, 50);
    imu_pub_     = nh_.advertise<nav_msgs::Odometry>(imu_odom_topic, 50);
    dbg_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(topic_prefix + "/debug/filtered_cloud", 1);

    // 初始化状态
    st_.pos = Eigen::Vector3d(init_x_, init_y_, init_z_);
    st_.vel.setZero();
    st_.prev_lin_acc.setZero();
    st_.has_prev = false;

    // 初始化预测位姿
    predict_pose_ = Eigen::Matrix4f::Identity();
    predict_pose_.block<3,1>(0,3) = Eigen::Vector3f(init_x_, init_y_, init_z_);

    ROS_INFO("IMU+LiDAR fusion initialized for drone_%d (skip_icp=%s, high_precision=%s)", 
             drone_id_, skip_icp_ ? "true" : "false", using_high_precision_ ? "true" : "false");
    ROS_INFO("Topics: IMU=%s, Fused=%s, IMU_odom=%s", 
             imu_topic.c_str(), fused_topic.c_str(), imu_odom_topic.c_str());
  }

private:
  // ===== ROS对象 =====
  ros::NodeHandle nh_;
  ros::Subscriber imu_sub_, cloud_sub_, map_sub_, odom_vis_sub_;
  ros::Publisher fused_pub_, imu_pub_, dbg_cloud_pub_;

  // ===== 参数 =====
  int drone_id_{0};
  float icp_max_distance_{1.0f};
  int icp_max_iter_{40};
  float voxel_leaf_size_{0.25f};
  int min_cloud_points_{1000};

  float init_x_{0}, init_y_{0}, init_z_{0};
  bool compensate_gravity_{true};
  double gravity_z_{9.80665};
  double fixed_dt_{-1.0}; // 若>=0，则使用固定dt（秒）
  bool skip_icp_{false}; // 快速调试开关
  bool using_high_precision_{false}; // 高精度模式开关

  std::string mode_;
  // ===== 状态与缓存 =====
  State st_;
  Eigen::Matrix4f predict_pose_{Eigen::Matrix4f::Identity()};

  // 地图
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
  pcl::PointCloud<pcl::PointNormal>::Ptr map_with_normals_;
  bool map_ready_{false};

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
    // 位置积分（与Python版本一致）
    st_.pos = integrate_vel(st_.pos, st_.vel, dt);
    st_.prev_lin_acc = lin_acc;
    st_.last_t = msg->header.stamp;

    // 更新预测位姿（旋转恒等）
    updatePredictPose();

    // 发布IMU积分结果
    publishIMUOdometry(msg->header.stamp);
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

  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    if (!map_ready_) {
      // 地图未准备好，直接发布IMU积分结果
      publishFusedOdometry(predict_pose_, msg->header.stamp, /*fallback=*/true);
      return;
    }

    // 快速调试：如果skip_icp_为true，直接使用IMU积分结果
    if (skip_icp_) {
      publishFusedOdometry(predict_pose_, msg->header.stamp, /*fallback=*/true);
      return;
    }

    // 转换点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *src);
    
    // 检查点云数量
    if ((int)src->size() < min_cloud_points_) {
      // 点数太少：直接用IMU积分结果
      publishFusedOdometry(predict_pose_, msg->header.stamp, /*fallback=*/true);
      return;
    }

    // 点云预处理：体素滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>(*src));
    voxelDownsample(filtered, voxel_leaf_size_);
    
    // 再次检查滤波后的点数
    if ((int)filtered->size() < min_cloud_points_) {
      publishFusedOdometry(predict_pose_, msg->header.stamp, /*fallback=*/true);
      return;
    }

    // 计算法线
    pcl::PointCloud<pcl::PointNormal>::Ptr src_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    computeNormals(filtered, src_with_normals);

    // ICP配准：使用IMU预测位姿作为初值
    Eigen::Matrix4f init_guess = predict_pose_;
    Eigen::Matrix4f T = runICP(src_with_normals, init_guess);

    // 检查ICP结果是否合理
    float trans_jump = (T.block<3,1>(0,3) - init_guess.block<3,1>(0,3)).norm();
    bool accept_icp = std::isfinite(T(0,0)) && (trans_jump < 2.0f);
    const Eigen::Matrix4f& final_pose = accept_icp ? T : init_guess;

    // 更新预测位姿（用于下一帧初值）
    predict_pose_ = final_pose;

    // 校正IMU状态：只校正位置，保持速度（避免跳变）
    st_.pos = final_pose.block<3,1>(0,3).cast<double>();

    // 发布融合结果
    publishFusedOdometry(final_pose, msg->header.stamp, !accept_icp);
  }

  void mapCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    if (map_ready_) return;

    pcl::fromROSMsg(*msg, *map_cloud_);
    voxelDownsample(map_cloud_, voxel_leaf_size_);

    // 预计算法线
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(map_cloud_);
    ne.setRadiusSearch(0.5);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);
    ne.compute(*normals);
    pcl::concatenateFields(*map_cloud_, *normals, *map_with_normals_);

    map_ready_ = true;
    ROS_INFO("Map ready with %zu points", map_cloud_->size());
  }

  // ===== 小工具函数 =====
  void voxelDownsample(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf, leaf, leaf);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
    vg.filter(*tmp);
    cloud.swap(tmp);

    sensor_msgs::PointCloud2 dbg;
    pcl::toROSMsg(*cloud, dbg);
    dbg.header.frame_id = "base_link";
    dbg_cloud_pub_.publish(dbg);
  }

  void computeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& xyz,
                      pcl::PointCloud<pcl::PointNormal>::Ptr& out) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(xyz);
    ne.setRadiusSearch(0.5);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr n(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*n);
    pcl::concatenateFields(*xyz, *n, *out);
  }

  Eigen::Matrix4f runICP(const pcl::PointCloud<pcl::PointNormal>::Ptr& src_with_normals,
                         const Eigen::Matrix4f& init_guess) {
    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
    icp.setInputSource(src_with_normals);
    icp.setInputTarget(map_with_normals_);
    icp.setMaxCorrespondenceDistance(icp_max_distance_);
    icp.setMaximumIterations(icp_max_iter_);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setUseReciprocalCorrespondences(true);

    pcl::PointCloud<pcl::PointNormal> aligned;
    icp.align(aligned, init_guess);

    if (!icp.hasConverged())
      return init_guess;
    return icp.getFinalTransformation();
  }

  void updatePredictPose() {
    // 只更新位置，旋转保持单位矩阵（不处理旋转）
    predict_pose_.setIdentity();
    predict_pose_.block<3,1>(0,3) = st_.pos.cast<float>();
  }


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
    ROS_INFO("Position: [%.2f, %.2f, %.2f]", st_.pos.x(), st_.pos.y(), st_.pos.z());
  }

  void publishFusedOdometry(const Eigen::Matrix4f& T, const ros::Time& stamp, bool fallback) {
    nav_msgs::Odometry od;
    od.header.stamp = stamp;
    od.header.frame_id = "map";
    od.child_frame_id = "base_link";

    // 位置（来自ICP或IMU积分）
    od.pose.pose.position.x = T(0,3);
    od.pose.pose.position.y = T(1,3);
    od.pose.pose.position.z = T(2,3);
    
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

    // 协方差：标记是否使用ICP结果（fallback时协方差更大）
    double pvar = fallback ? 0.5 : 0.05;
    od.pose.covariance[0] = od.pose.covariance[7] = od.pose.covariance[14] = pvar;

    fused_pub_.publish(od);
  }
};

#endif // VISION_IMU_FUSION_HPP