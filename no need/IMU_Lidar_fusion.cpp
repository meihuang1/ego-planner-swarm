#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

class IMULidarFusion
{
public:
  IMULidarFusion(ros::NodeHandle &nh)
  {
    // 订阅全局地图点云
    cloud_map_sub = nh.subscribe("/map_generator/global_cloud", 1, &IMULidarFusion::mapCallback, this);

    // 订阅IMU和雷达点云
    imu_sub = nh.subscribe("/drone_0/imu", 100, &IMULidarFusion::imuCallback, this);
    cloud_sub = nh.subscribe("/drone_0_pcl_render_node/cloud", 10, &IMULidarFusion::cloudCallback, this);

    odom_pub = nh.advertise<nav_msgs::Odometry>("/fusion/odom", 10);

    predict_pose = Eigen::Matrix4f::Identity();
    predict_pose(0, 3) = -20.0f; // 初始位置x
    predict_pose(1, 3) = -9.0f;  // 初始位置y
    predict_pose(2, 3) = 1.0f;   // 初始位置z

    orientation = Eigen::Quaternionf::Identity();
    velocity.setZero();
    position.setZero();

    map_ready = false;
  }

  void mapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
  {
    pcl::fromROSMsg(*msg, *map_cloud);

    pcl::VoxelGrid<pcl::PointXYZ> voxel_map;
    voxel_map.setLeafSize(0.3f, 0.3f, 0.3f);
    voxel_map.setInputCloud(map_cloud);
    voxel_map.filter(*map_cloud);

    map_ready = true;
    ROS_INFO("Map received and filtered. Points: %zu", map_cloud->size());
  }

  void imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
  {
    if (last_imu_time.toSec() == 0)
    {
      last_imu_time = msg->header.stamp;
      return;
    }

    double dt = (msg->header.stamp - last_imu_time).toSec();
    last_imu_time = msg->header.stamp;

    Eigen::Vector3f acc(msg->linear_acceleration.x,
                        msg->linear_acceleration.y,
                        msg->linear_acceleration.z);
    Eigen::Vector3f gyro(msg->angular_velocity.x,
                         msg->angular_velocity.y,
                         msg->angular_velocity.z);

    // 计算旋转增量
    Eigen::Vector3f delta_angle = gyro * dt;
    float angle = delta_angle.norm();
    Eigen::Quaternionf delta_q = Eigen::Quaternionf::Identity();
    if (angle > 1e-5)
    {
      delta_q = Eigen::AngleAxisf(angle, delta_angle.normalized());
    }

    orientation = (orientation * delta_q).normalized();

    // 假设重力朝向地图Z轴负方向
    Eigen::Vector3f gravity(0, 0, -9.8f);
    Eigen::Vector3f acc_corrected = orientation * acc + gravity;

    velocity += acc_corrected * dt;
    position += velocity * dt + 0.5f * acc_corrected * dt * dt;

    predict_pose = Eigen::Matrix4f::Identity();
    predict_pose.block<3, 3>(0, 0) = orientation.toRotationMatrix().cast<float>();
    predict_pose.block<3, 1>(0, 3) = position.cast<float>();
  }
void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  ROS_INFO_STREAM("predict_pose translation: "
                  << predict_pose(0, 3) << ", " << predict_pose(1, 3) << ", " << predict_pose(2, 3));

  if (!map_ready || map_cloud->empty())
  {
    ROS_WARN("Map not ready, skipping ICP alignment...");
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *input);

  if (input->empty())
  {
    ROS_WARN("Input cloud empty, skipping");
    return;
  }

  // 下采样减少计算
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);
  voxel_filter.setInputCloud(input);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_input(new pcl::PointCloud<pcl::PointXYZ>);
  voxel_filter.filter(*filtered_input);

  // === 1. 计算当前帧法线 ===
  pcl::PointCloud<pcl::Normal>::Ptr normals_input(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_input;
  ne_input.setInputCloud(filtered_input);
  ne_input.setRadiusSearch(0.5); // 经验值：越稠密越小
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_input(new pcl::search::KdTree<pcl::PointXYZ>);
  ne_input.setSearchMethod(tree_input);
  ne_input.compute(*normals_input);

  // 合并点云 + 法线
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_input_with_normals(new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields(*filtered_input, *normals_input, *cloud_input_with_normals);

  // === 2. 地图也做法线处理（建议预处理好存着） ===
  static pcl::PointCloud<pcl::PointNormal>::Ptr map_with_normals(new pcl::PointCloud<pcl::PointNormal>);
  static bool map_normals_ready = false;
  if (!map_normals_ready)
  {
    pcl::PointCloud<pcl::Normal>::Ptr map_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_map;
    ne_map.setInputCloud(map_cloud);
    ne_map.setRadiusSearch(0.5);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_map(new pcl::search::KdTree<pcl::PointXYZ>);
    ne_map.setSearchMethod(tree_map);
    ne_map.compute(*map_normals);

    pcl::concatenateFields(*map_cloud, *map_normals, *map_with_normals);
    map_normals_ready = true;
  }

  // === 3. 使用带法线的 ICP ===
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
  icp.setInputSource(cloud_input_with_normals);
  icp.setInputTarget(map_with_normals);

  icp.setMaxCorrespondenceDistance(20.0);
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(1e-8);
  icp.setEuclideanFitnessEpsilon(1e-6);

  Eigen::Matrix4f init_guess = predict_pose.allFinite() ? predict_pose : last_transform;

  pcl::PointCloud<pcl::PointNormal> aligned;
  try
  {
    icp.align(aligned, init_guess);
  }
  catch (const std::exception &e)
  {
    ROS_ERROR("ICP failed: %s", e.what());
    return;
  }

  if (!icp.hasConverged())
  {
    ROS_WARN("ICP did not converge, keeping last pose");
    return;
  }

  Eigen::Matrix4f result = icp.getFinalTransformation();

  publishOdometry(result);

  last_transform = result;
  predict_pose = result;

  position = result.block<3, 1>(0, 3);
  orientation = Eigen::Quaternionf(result.block<3, 3>(0, 0));

  ROS_INFO_STREAM("ICP Fitness Score: " << icp.getFitnessScore());
  ROS_INFO_STREAM("ICP Result translation: "
                  << result(0, 3) << ", " << result(1, 3) << ", " << result(2, 3));
}


  void publishOdometry(const Eigen::Matrix4f &pose)
  {
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_link";

    odom_msg.pose.pose.position.x = pose(0, 3);
    odom_msg.pose.pose.position.y = pose(1, 3);
    odom_msg.pose.pose.position.z = pose(2, 3);

    Eigen::Quaternionf q(pose.block<3, 3>(0, 0));
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();

    odom_pub.publish(odom_msg);
  }

private:
  ros::Subscriber imu_sub, cloud_sub, cloud_map_sub;
  ros::Publisher odom_pub;

  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud{new pcl::PointCloud<pcl::PointXYZ>};

  Eigen::Matrix4f predict_pose;
  Eigen::Matrix4f last_transform = Eigen::Matrix4f::Identity();

  ros::Time last_imu_time;

  Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();

  bool map_ready;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "imu_lidar_fusion_node");
  ros::NodeHandle nh;

  IMULidarFusion fusion_node(nh);

  ros::spin();
  return 0;
}
