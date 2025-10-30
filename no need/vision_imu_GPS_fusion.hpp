#ifndef VISION_IMU_GPS_FUSION_HPP
#define VISION_IMU_GPS_FUSION_HPP

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <Eigen/Dense>

// ---------- IMU状态结构体 ----------
struct State
{
  Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ang_vel{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond quat{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d prev_lin_acc{Eigen::Vector3d::Zero()};
  Eigen::Vector3d prev_ang_acc{Eigen::Vector3d::Zero()};
  bool has_prev{false};
  ros::Time last_t;
};

// ---------- 数学/工具函数 ----------

// 梯形积分：加速度->速度
inline Eigen::Vector3d integrate_lin_acc(const Eigen::Vector3d &prev_vel,
                                         const Eigen::Vector3d &lin_acc,
                                         const Eigen::Vector3d &prev_lin_acc,
                                         double dt)
{
  return prev_vel + 0.5 * (prev_lin_acc + lin_acc) * dt;
}

// 速度->位置
inline Eigen::Vector3d integrate_vel(const Eigen::Vector3d &prev_pos,
                                     const Eigen::Vector3d &vel,
                                     double dt)
{
  return prev_pos + vel * dt;
}

// 梯形积分：角加速度->角速度
inline Eigen::Vector3d integrate_ang_acc(const Eigen::Vector3d &prev_ang_vel,
                                         const Eigen::Vector3d &ang_acc,
                                         const Eigen::Vector3d &prev_ang_acc,
                                         double dt)
{
  return prev_ang_vel + 0.5 * (prev_ang_acc + ang_acc) * dt;
}

// 角速度积分->姿态（四元数）
inline Eigen::Quaterniond integrate_ang_vel(const Eigen::Quaterniond &prev_quat,
                                            const Eigen::Vector3d &ang_vel,
                                            double dt)
{
  double omega_mag = ang_vel.norm();
  Eigen::Quaterniond dq;
  if (omega_mag * dt < 1e-8)
  {
    // 小角度近似
    Eigen::Vector3d half = 0.5 * ang_vel * dt;
    dq = Eigen::Quaterniond(1.0, half.x(), half.y(), half.z());
  }
  else
  {
    double theta = omega_mag * dt;
    Eigen::Vector3d axis = ang_vel / omega_mag;
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    dq = Eigen::Quaterniond(c, axis.x() * s, axis.y() * s, axis.z() * s);
  }
  Eigen::Quaterniond q_new = prev_quat * dq;
  q_new.normalize();
  return q_new;
}

class IMUVisionGPSFusion
{
public:
  IMUVisionGPSFusion(ros::NodeHandle &nh) : nh_(nh)
  {
    // 获取drone_id参数，默认为0
    nh_.param<int>("drone_id", drone_id_, 0);

    // 加载ICP参数
    nh_.param<float>("icp_max_distance", icp_max_distance_, 1.0);
    nh_.param<int>("icp_max_iter", icp_max_iter_, 40);
    nh_.param<float>("voxel_leaf_size", voxel_leaf_size_, 0.25);

    nh_.param<float>("init_x", init_x_, 0.0);
    nh_.param<float>("init_y", init_y_, 0.0);
    nh_.param<float>("init_z", init_z_, 0.0);
    // 构建话题前缀
    std::string topic_prefix = "/drone_" + std::to_string(drone_id_);

    // 初始化点云
    map_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    map_with_normals.reset(new pcl::PointCloud<pcl::PointNormal>);

    // 订阅全局地图点云（不加前缀）
    cloud_map_sub = nh.subscribe("/map_generator/global_cloud", 1, &IMUVisionGPSFusion::mapCallback, this);

    // 使用message_filters同步里程计和点云（加前缀）
    odom_sub.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh, topic_prefix + "/odom_fused_IG_global", 10));
    cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, topic_prefix + "_pcl_render_node/cloud", 10));

    sync.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *odom_sub, *cloud_sub));
    sync->registerCallback(boost::bind(&IMUVisionGPSFusion::syncCloudOdomCallback, this, _1, _2));

    // 订阅IMU用于预测（加前缀）
    imu_sub = nh.subscribe(topic_prefix + "/imu", 100, &IMUVisionGPSFusion::imuCallback, this);

    // 发布器（加前缀）
    odom_pub = nh.advertise<nav_msgs::Odometry>(topic_prefix + "/odom_fused_IGV", 10);
    imu_odom_pub = nh.advertise<nav_msgs::Odometry>(topic_prefix + "/imu_integrator_res2", 10);

    debug_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(topic_prefix + "/debug/filtered_cloud", 1);
    debug_map_pub = nh.advertise<sensor_msgs::PointCloud2>(topic_prefix + "/debug/filtered_map_cloud", 1);

    // 初始化参数
    predict_pose = Eigen::Matrix4f::Identity();
    predict_pose.block<3, 1>(0, 3) = Eigen::Vector3f(init_x_, init_y_, init_z_);
    last_transform = Eigen::Matrix4f::Identity();
    map_ready = false;

    // 初始化重力补偿参数
    gravity_z_ = 9.80665;
    compensate_gravity_ = true;

    ROS_INFO("IMU-LiDAR Fusion initialized for drone_%d", drone_id_);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber imu_sub, cloud_map_sub;
  std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
  ros::Publisher odom_pub, imu_odom_pub, debug_cloud_pub, debug_map_pub;

  int drone_id_;
  // 定义同步策略
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> SyncPolicy;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync;

  // 点云相关
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud;
  pcl::PointCloud<pcl::PointNormal>::Ptr map_with_normals;
  bool map_ready;

  // 位姿相关
  Eigen::Matrix4f predict_pose;
  Eigen::Matrix4f last_transform;

  // ICP参数
  float icp_max_distance_;
  int icp_max_iter_;
  float voxel_leaf_size_;
  float init_x_, init_y_, init_z_;

  // IMU相关
  double gravity_z_;
  bool compensate_gravity_;
  State state_;

  // ---------- 主要回调函数 ----------
  void syncCloudOdomCallback(const nav_msgs::Odometry::ConstPtr &odom_msg,
                             const sensor_msgs::PointCloud2::ConstPtr &cloud_msg)
  {
    if (!map_ready)
    {
      ROS_WARN_THROTTLE(1.0, "Map not ready, skipping ICP");
      return;
    }

    // 转换点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *input);

    // if (input->empty())
    // {
    //   ROS_WARN("Empty input cloud");
    //   return;
    // }

    // 预处理点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = preprocessCloudUnified(input, voxel_leaf_size_, false);

    // 发布调试点云
    publishDebugClouds(filtered);

    // 将odom_msg转换为Eigen::Matrix4f作为初始猜测
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

    // 设置位置
    init_guess.block<3, 1>(0, 3) = Eigen::Vector3f(
        odom_msg->pose.pose.position.x,
        odom_msg->pose.pose.position.y,
        odom_msg->pose.pose.position.z);

    // 设置姿态（四元数转旋转矩阵）
    Eigen::Quaternionf q(
        odom_msg->pose.pose.orientation.w,
        odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y,
        odom_msg->pose.pose.orientation.z);
    q.normalize();
    init_guess.block<3, 3>(0, 0) = q.toRotationMatrix();

    // 执行ICP配准
    Eigen::Matrix4f icp_result = runICP(filtered, init_guess);

    // 智能选择最终位姿（现在使用odom作为参考）
    Eigen::Matrix4f final_pose = selectFinalPose(init_guess, icp_result);

    // 发布最终结果
    publishOdometry(final_pose);

    // 更新状态
    updateState(final_pose);
  }

  void imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
  {
    const ros::Time t = msg->header.stamp;
    if (!state_.has_prev)
    {
      initializeState(msg);
      return;
    }

    // const double dt = (t - state_.last_t).toSec();
    const double dt = 0.01;
    if (dt <= 0.0)
      return;

    // 处理IMU数据
    processIMUData(msg, dt);

    // 更新predict_pose
    updatePredictPose();

    // 发布IMU积分结果
    publishIMUOdometry(msg);

    state_.last_t = t;
  }

  // ---------- 辅助函数 ----------

  void initializeState(const sensor_msgs::Imu::ConstPtr &msg)
  {
    const auto &q = msg->orientation;
    if (!(std::isnan(q.w) || std::isnan(q.x) || std::isnan(q.y) || std::isnan(q.z)))
    {
      state_.quat = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
      state_.quat.normalize();
    }
    else
    {
      state_.quat.setIdentity();
    }

    // 使用参数化的初始位置
    state_.pos = Eigen::Vector3d(init_x_, init_y_, init_z_);

    state_.prev_lin_acc.setZero();
    state_.prev_ang_acc.setZero();
    state_.last_t = msg->header.stamp;
    state_.has_prev = true;

    // 同时更新predict_pose
    updatePredictPose();

    ROS_INFO("Initialized state at position (%.2f, %.2f, %.2f)", init_x_, init_y_, init_z_);
  }

  void processIMUData(const sensor_msgs::Imu::ConstPtr &msg, double dt)
  {
    // 读取线加速度并补偿重力
    Eigen::Vector3d lin_acc(msg->linear_acceleration.x,
                            msg->linear_acceleration.y,
                            msg->linear_acceleration.z);
    if (compensate_gravity_)
    {
      lin_acc.z() -= gravity_z_;
    }

    // 读取角速度
    Eigen::Vector3d ang_acc(msg->angular_velocity.x,
                            msg->angular_velocity.y,
                            msg->angular_velocity.z);

    // 积分更新状态
    state_.vel = integrate_lin_acc(state_.vel, lin_acc, state_.prev_lin_acc, dt);
    state_.pos = integrate_vel(state_.pos, state_.vel, dt);
    state_.prev_lin_acc = lin_acc;

    state_.ang_vel = integrate_ang_acc(state_.ang_vel, ang_acc, state_.prev_ang_acc, dt);
    state_.prev_ang_acc = ang_acc;

    state_.quat = integrate_ang_vel(state_.quat, state_.ang_vel, dt);
  }

  void updatePredictPose()
  {
    predict_pose.block<3, 1>(0, 3) = state_.pos.cast<float>();
    predict_pose.block<3, 3>(0, 0) = state_.quat.toRotationMatrix().cast<float>();
  }

  void publishIMUOdometry(const sensor_msgs::Imu::ConstPtr &msg)
  {
    nav_msgs::Odometry odom;
    odom.header.stamp = msg->header.stamp;
    odom.header.frame_id = "world";
    odom.child_frame_id = "base_link";

    odom.pose.pose.position.x = state_.pos.x();
    odom.pose.pose.position.y = state_.pos.y();
    odom.pose.pose.position.z = state_.pos.z();
    odom.pose.pose.orientation.x = state_.quat.x();
    odom.pose.pose.orientation.y = state_.quat.y();
    odom.pose.pose.orientation.z = state_.quat.z();
    odom.pose.pose.orientation.w = state_.quat.w();

    odom.twist.twist.linear.x = state_.vel.x();
    odom.twist.twist.linear.y = state_.vel.y();
    odom.twist.twist.linear.z = state_.vel.z();
    odom.twist.twist.angular.x = state_.ang_vel.x();
    odom.twist.twist.angular.y = state_.ang_vel.y();
    odom.twist.twist.angular.z = state_.ang_vel.z();

    imu_odom_pub.publish(odom);
  }

  void publishDebugClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered)
  {
    sensor_msgs::PointCloud2 debug_msg;
    pcl::toROSMsg(*filtered, debug_msg);
    debug_msg.header.frame_id = "base_link";
    // debug_cloud_pub.publish(debug_msg);

    sensor_msgs::PointCloud2 ros_map_cloud;
    pcl::toROSMsg(*map_cloud, ros_map_cloud);
    ros_map_cloud.header.frame_id = "world";
    // debug_map_pub.publish(ros_map_cloud);
  }

  Eigen::Matrix4f selectFinalPose(const Eigen::Matrix4f &imu_prediction, const Eigen::Matrix4f &icp_result)
  {
    // 比较ICP结果与IMU预测的差异
    Eigen::Vector3f imu_pos = imu_prediction.block<3, 1>(0, 3);
    Eigen::Vector3f icp_pos = icp_result.block<3, 1>(0, 3);
    float position_diff = (icp_pos - imu_pos).norm();

    Eigen::Quaternionf imu_quat(imu_prediction.block<3, 3>(0, 0));
    Eigen::Quaternionf icp_quat(icp_result.block<3, 3>(0, 0));
    float orientation_diff = std::abs(imu_quat.angularDistance(icp_quat));

    // 设置阈值
    float position_threshold = 2.0f;
    float orientation_threshold = 0.3f;

    if (position_diff < position_threshold && orientation_diff < orientation_threshold)
    {
      ROS_INFO("ICP result accepted: pos_diff=%.2fm, ori_diff=%.2frad, using ICP",
               position_diff, orientation_diff);
      return icp_result;
    }
    else
    {
      ROS_WARN("ICP result rejected: pos_diff=%.2fm, ori_diff=%.2frad, using IMU",
               position_diff, orientation_diff);
      return imu_prediction;
    }
  }

  void updateState(const Eigen::Matrix4f &final_pose)
  {
    last_transform = final_pose;
    predict_pose = final_pose;

    Eigen::Vector3f position = final_pose.block<3, 1>(0, 3);
    Eigen::Quaternionf orientation(final_pose.block<3, 3>(0, 0));

    ROS_INFO("Final pose: pos=(%.2f, %.2f, %.2f)",
             position.x(), position.y(), position.z());
  }

  // ---------- 点云处理函数 ----------

  pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessCloudUnified(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &input,
      float voxel_size,
      bool apply_statistical_filter = true)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>(*input));

    if (apply_statistical_filter)
    {
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud(filtered);
      sor.setMeanK(50);
      sor.setStddevMulThresh(1.0);
      sor.filter(*filtered);
    }

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(filtered);
    voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel.filter(*filtered);

    return filtered;
  }

  Eigen::Matrix4f runICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, const Eigen::Matrix4f &init_guess)
  {
    // 计算输入点云法线
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(input);
    ne.setRadiusSearch(0.5);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);
    ne.compute(*normals);

    // 合并点云和法线
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*input, *normals, *cloud_with_normals);

    // 配置ICP
    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
    icp.setInputSource(cloud_with_normals);
    icp.setInputTarget(map_with_normals);
    icp.setMaxCorrespondenceDistance(icp_max_distance_);
    icp.setMaximumIterations(icp_max_iter_);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setUseReciprocalCorrespondences(true);

    // 执行ICP
    pcl::PointCloud<pcl::PointNormal> aligned;
    icp.align(aligned, init_guess);

    if (!icp.hasConverged())
    {
      // ROS_WARN("ICP did not converge, using odometry pose");
      return init_guess;
    }

    // ROS_INFO_STREAM("ICP fitness: " << icp.getFitnessScore()
    //                                 << ", iterations: " << icp.getMaximumIterations());

    // 检查ICP结果是否合理
    float translation_diff = (icp.getFinalTransformation().block<3, 1>(0, 3) -
                              init_guess.block<3, 1>(0, 3))
                                 .norm();
    if (translation_diff > 2.0)
    {
      // ROS_WARN("Large ICP deviation (%.2fm), using odometry pose", translation_diff);
      return init_guess;
    }

    return icp.getFinalTransformation();
  }

  void mapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
  {
    pcl::fromROSMsg(*msg, *map_cloud);
    map_cloud = preprocessCloudUnified(map_cloud, voxel_leaf_size_, false);

    // 预计算地图法线
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(map_cloud);
    ne.setRadiusSearch(0.5);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr map_normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*map_normals);
    pcl::concatenateFields(*map_cloud, *map_normals, *map_with_normals);

    map_ready = true;
    ROS_INFO("Map ready with %zu points", map_cloud->size());
  }

  void publishOdometry(const Eigen::Matrix4f &pose)
  {
    nav_msgs::Odometry odom;
    odom.header.stamp = ros::Time::now();
    odom.header.frame_id = "map";
    odom.child_frame_id = "base_link";

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);

    Eigen::Quaternionf q(pose.block<3, 3>(0, 0));
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x = state_.vel.x();
    odom.twist.twist.linear.y = state_.vel.y();
    odom.twist.twist.linear.z = state_.vel.z();
    odom.twist.twist.angular.x = state_.ang_vel.x();
    odom.twist.twist.angular.y = state_.ang_vel.y();
    odom.twist.twist.angular.z = state_.ang_vel.z();

    odom_pub.publish(odom);
  }
};

#endif