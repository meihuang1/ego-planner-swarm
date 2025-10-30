#include <iostream>
#include <fstream>
#include <vector>
// ros
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// eigen / opencv
#include <Eigen/Eigen>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include "depth_render.cuh"

using namespace std;
using namespace Eigen;

// CUDA depth buffer host ptr
int *depth_hostptr;
cv::Mat depth_mat;

// camera intrinsics
int width, height;
double fx, fy, cx, cy;

// renderer
DepthRender depthrender;

// pubs
ros::Publisher pub_depth;
ros::Publisher pub_color;
ros::Publisher pub_pose;
ros::Publisher pub_pcl_world;

// msgs
sensor_msgs::PointCloud2 local_map_pcl;

// subs
ros::Subscriber odom_sub;
ros::Subscriber global_map_sub, local_map_sub;

// timers
ros::Timer local_sensing_timer, estimation_timer;

// flags
bool has_global_map(false);
bool has_local_map(false);
bool has_odom(false);

// poses
Matrix4d cam2world;
Quaterniond cam2world_quat;
nav_msgs::Odometry _odom;

// params
double sensing_horizon, sensing_rate, estimation_rate;
double topdown_height;            // camera height above target point

// map params (for grid helpers if needed)
double _x_size, _y_size, _z_size;
double _gl_xl, _gl_yl, _gl_zl;
double _resolution = 0.1, _inv_resolution = 10.0;
int _GLX_SIZE, _GLY_SIZE, _GLZ_SIZE;

ros::Time last_odom_stamp = ros::TIME_MAX;
Vector3d last_pose_world(0.0, 0.0, 0.0);

inline Vector3d gridIndex2coord(const Vector3i & index)
{
  Vector3d pt;
  pt(0) = ((double)index(0) + 0.5) * _resolution + _gl_xl;
  pt(1) = ((double)index(1) + 0.5) * _resolution + _gl_yl;
  pt(2) = ((double)index(2) + 0.5) * _resolution + _gl_zl;
  return pt;
}

inline Vector3i coord2gridIndex(const Vector3d & pt)
{
  Vector3i idx;
  idx(0) = std::min( std::max( int( (pt(0) - _gl_xl) * _inv_resolution), 0), _GLX_SIZE - 1);
  idx(1) = std::min( std::max( int( (pt(1) - _gl_yl) * _inv_resolution), 0), _GLY_SIZE - 1);
  idx(2) = std::min( std::max( int( (pt(2) - _gl_zl) * _inv_resolution), 0), _GLZ_SIZE - 1);
  return idx;
}


void rcvOdometryCallbck(const nav_msgs::Odometry& odom)
{
  has_odom = true;
  _odom = odom;

  // Build top-down camera pose: position at (odom.x, odom.y, odom.z + topdown_height)
  Vector3d center_xy(_odom.pose.pose.position.x, _odom.pose.pose.position.y, _odom.pose.pose.position.z);
  Vector3d cam_pos = center_xy;
  cam_pos.z() += topdown_height;  // place camera high above

  // Orientation: look straight down (-Z world) with 90 degree counter-clockwise rotation around Z
  Matrix3d R = Matrix3d::Identity();
  // 绕Z轴逆时针旋转90度: cos(-90°)=0, sin(-90°)=-1
  R(0,0) = 0.0;  R(0,1) = -1.0; R(0,2) = 0.0;  // X轴 -> -Y轴方向
  R(1,0) = 1.0;  R(1,1) = 0.0;  R(1,2) = 0.0;  // Y轴 -> X轴方向  
  R(2,0) = 0.0;  R(2,1) = 0.0;  R(2,2) = -1.0; // Z轴 -> -Z轴方向(向下看)

  cam2world.setIdentity();
  cam2world.block<3,3>(0,0) = R;
  cam2world.block<3,1>(0,3) = cam_pos;
  cam2world_quat = cam2world.block<3,3>(0,0);
  last_odom_stamp = odom.header.stamp;
  last_pose_world = center_xy;
}

vector<float> cloud_data;

void rcvGlobalPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map)
{
  if (has_global_map) return;
  ROS_WARN("Global Pointcloud received..");
  pcl::PointCloud<pcl::PointXYZ> cloudIn;
  pcl::fromROSMsg(pointcloud_map, cloudIn);
  cloud_data.reserve(cloudIn.points.size() * 3);
  for (size_t i = 0; i < cloudIn.points.size(); i++) {
    const auto &pt = cloudIn.points[i];
    cloud_data.push_back(pt.x);
    cloud_data.push_back(pt.y);
    cloud_data.push_back(pt.z);
  }
  depthrender.set_data(cloud_data);
  depth_hostptr = (int*) malloc(width * height * sizeof(int));
  has_global_map = true;
}

void rcvLocalPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map)
{
  pcl::PointCloud<pcl::PointXYZ> cloudIn;
  pcl::fromROSMsg(pointcloud_map, cloudIn);
  if (cloudIn.points.empty()) return;
  for (size_t i = 0; i < cloudIn.points.size(); i++) {
    const auto &pt = cloudIn.points[i];
    cloud_data.push_back(pt.x);
    cloud_data.push_back(pt.y);
    cloud_data.push_back(pt.z);
  }
  depthrender.set_data(cloud_data);
  if (!depth_hostptr) depth_hostptr = (int*) malloc(width * height * sizeof(int));
  has_local_map = true;
}

void render_pcl_world()
{
  // compose point cloud from depth image
  pcl::PointCloud<pcl::PointXYZ> localMap;
  pcl::PointXYZ pt_in;
  Vector4d pose_in_camera;
  Vector4d pose_in_world;
  Vector3d pose_pt;

  for (int u = 0; u < width; u++)
    for (int v = 0; v < height; v++) {
      float depth = depth_mat.at<float>(v,u);
      if (depth == 0.0f) continue;
      pose_in_camera(0) = (u - cx) * depth / fx;
      pose_in_camera(1) = (v - cy) * depth / fy;
      pose_in_camera(2) = depth;
      pose_in_camera(3) = 1.0;
      pose_in_world = cam2world * pose_in_camera;

      // 移除sensing_horizon限制，让点云范围与深度图一致
      // if ((pose_in_world.segment(0,3) - last_pose_world).norm() > sensing_horizon)
      //   continue;

      pose_pt = pose_in_world.head(3);
      pt_in.x = pose_pt(0);
      pt_in.y = pose_pt(1);
      pt_in.z = pose_pt(2);
      localMap.points.push_back(pt_in);
    }

  localMap.width = localMap.points.size();
  localMap.height = 1;
  localMap.is_dense = true;

  pcl::toROSMsg(localMap, local_map_pcl);
  local_map_pcl.header.frame_id  = "map";
  local_map_pcl.header.stamp     = last_odom_stamp;
  pub_pcl_world.publish(local_map_pcl);
}

void render_currentpose()
{
  Matrix4d cam_pose = cam2world.inverse();

  double pose[4 * 4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      pose[j + 4 * i] = cam_pose(i, j);

  depthrender.render_pose(pose, depth_hostptr);

  depth_mat = cv::Mat::zeros(height, width, CV_32FC1);
  double min = 0.5;
  double max = 1.0f;
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      float depth = (float)depth_hostptr[i * width + j] / 1000.0f;
      depth = depth < 500.0f ? depth : 0;
      max = depth > max ? depth : max;
      depth_mat.at<float>(i,j) = depth;
    }

  // 调试信息：统计有效深度像素
  int valid_pixels = 0;
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      if (depth_mat.at<float>(i,j) > 0.0f) valid_pixels++;
    }
  ROS_INFO("Depth image: %d valid pixels out of %dx%d total", 
           valid_pixels, width, height);

  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = last_odom_stamp;
  out_msg.header.frame_id = "camera";
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  out_msg.image = depth_mat.clone();
  pub_depth.publish(out_msg.toImageMsg());

  cv::Mat adjMap;
  depth_mat.convertTo(adjMap, CV_8UC1, 255 / 13.0, -min);
  cv::Mat falseColorsMap;
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
  cv_bridge::CvImage cv_image_colored;
  cv_image_colored.header.frame_id = "depthmap";
  cv_image_colored.header.stamp = last_odom_stamp;
  cv_image_colored.encoding = sensor_msgs::image_encodings::BGR8;
  cv_image_colored.image = falseColorsMap;
  pub_color.publish(cv_image_colored.toImageMsg());
}

void pubCameraPose(const ros::TimerEvent &)
{
  geometry_msgs::PoseStamped camera_pose;
  camera_pose.header = _odom.header;
  camera_pose.header.frame_id = "/map";
  camera_pose.pose.position.x = cam2world(0,3);
  camera_pose.pose.position.y = cam2world(1,3);
  camera_pose.pose.position.z = cam2world(2,3);
  camera_pose.pose.orientation.w = cam2world_quat.w();
  camera_pose.pose.orientation.x = cam2world_quat.x();
  camera_pose.pose.orientation.y = cam2world_quat.y();
  camera_pose.pose.orientation.z = cam2world_quat.z();
  pub_pose.publish(camera_pose);
}

void renderSensedPoints(const ros::TimerEvent &)
{
  if (!has_odom) return;
  if (!has_global_map && !has_local_map) return;
  render_currentpose();
  render_pcl_world();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pcl_render_topdown");
  ros::NodeHandle nh("~");

  // camera intrinsics / image size
  nh.getParam("cam_width", width);
  nh.getParam("cam_height", height);
  nh.getParam("cam_fx", fx);
  nh.getParam("cam_fy", fy);
  nh.getParam("cam_cx", cx);
  nh.getParam("cam_cy", cy);
  nh.getParam("sensing_horizon", sensing_horizon);
  nh.getParam("sensing_rate", sensing_rate);
  nh.getParam("estimation_rate", estimation_rate);

  // topdown params
  nh.param("topdown_height", topdown_height, -20.0);

  nh.getParam("map/x_size", _x_size);
  nh.getParam("map/y_size", _y_size);
  nh.getParam("map/z_size", _z_size);

  depthrender.set_para(fx, fy, cx, cy, width, height);

  // init pose
  cam2world = Matrix4d::Identity();

  // subs
  global_map_sub = nh.subscribe("global_map", 1, rcvGlobalPointCloudCallBack);
  local_map_sub  = nh.subscribe("local_map",  1, rcvLocalPointCloudCallBack);
  odom_sub       = nh.subscribe("odometry",   50, rcvOdometryCallbck);

  // pubs
  pub_depth = nh.advertise<sensor_msgs::Image>("depth", 10);
  pub_color = nh.advertise<sensor_msgs::Image>("colordepth", 10);
  pub_pose  = nh.advertise<geometry_msgs::PoseStamped>("camera_pose", 10);
  pub_pcl_world = nh.advertise<sensor_msgs::PointCloud2>("rendered_pcl", 1);

  double sensing_duration  = 1.0 / sensing_rate;
  double estimate_duration = 1.0 / estimation_rate;
  local_sensing_timer = nh.createTimer(ros::Duration(sensing_duration),  renderSensedPoints);
  estimation_timer    = nh.createTimer(ros::Duration(estimate_duration), pubCameraPose);

  _inv_resolution = 1.0 / _resolution;
  _gl_xl = -_x_size/2.0; _gl_yl = -_y_size/2.0; _gl_zl = 0.0;
  _GLX_SIZE = (int)(_x_size * _inv_resolution);
  _GLY_SIZE = (int)(_y_size * _inv_resolution);
  _GLZ_SIZE = (int)(_z_size * _inv_resolution);

  ros::Rate rate(100);
  while (ros::ok()) {
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}

