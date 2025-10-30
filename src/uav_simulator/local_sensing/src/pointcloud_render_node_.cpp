#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <pcl/search/impl/kdtree.hpp>
#include <vector>
#include <plan_env/raycast.h>

struct PointXYZIRT
{
  PCL_ADD_POINT4D;
  float intensity;
  std::uint16_t ring;
  float time; // ✅ 新增的字段：时间戳
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(float, time, time)) // ✅ 注册 time 字段

using namespace std;
using namespace Eigen;

ros::Publisher pub_cloud;
ros::Publisher pub_occupancy_buffer;

sensor_msgs::PointCloud2 local_map_pcl;
sensor_msgs::PointCloud2 local_depth_pcl;

ros::Subscriber odom_sub;
ros::Subscriber global_map_sub, local_map_sub;

ros::Timer local_sensing_timer;

bool has_global_map(false);
bool has_local_map(false);
bool has_odom(false);

nav_msgs::Odometry _odom;

double sensing_horizon, sensing_rate, estimation_rate;
double _x_size, _y_size, _z_size;
double _gl_xl, _gl_yl, _gl_zl;
double _resolution, _inv_resolution;
int _GLX_SIZE, _GLY_SIZE, _GLZ_SIZE;

ros::Time last_odom_stamp = ros::TIME_MAX;

inline Eigen::Vector3d gridIndex2coord(const Eigen::Vector3i &index)
{
  Eigen::Vector3d pt;
  pt(0) = ((double)index(0) + 0.5) * _resolution + _gl_xl;
  pt(1) = ((double)index(1) + 0.5) * _resolution + _gl_yl;
  pt(2) = ((double)index(2) + 0.5) * _resolution + _gl_zl;

  return pt;
};

inline Eigen::Vector3i coord2gridIndex(const Eigen::Vector3d &pt)
{
  Eigen::Vector3i idx;

  idx(0) = std::min(std::max(int((pt(0) - _gl_xl) * _inv_resolution), 0),
                    _GLX_SIZE - 1);
  idx(1) = std::min(std::max(int((pt(1) - _gl_yl) * _inv_resolution), 0),
                    _GLY_SIZE - 1);
  idx(2) = std::min(std::max(int((pt(2) - _gl_zl) * _inv_resolution), 0),
                    _GLZ_SIZE - 1);
  assert(idx(0) >= 0 && idx(0) < _GLX_SIZE);
  assert(idx(1) >= 0 && idx(1) < _GLY_SIZE);
  assert(idx(2) >= 0 && idx(2) < _GLZ_SIZE);
  return idx;
};

void rcvOdometryCallbck(const nav_msgs::Odometry &odom)
{
  /*if(!has_global_map)
    return;*/
  has_odom = true;
  _odom = odom;
}

pcl::PointCloud<pcl::PointXYZ> _cloud_all_map;

pcl::PointCloud<PointXYZIRT> _local_map;

std::vector<int> occupancy_buffer; // 全局变量

pcl::VoxelGrid<pcl::PointXYZ> _voxel_sampler;
sensor_msgs::PointCloud2 _local_map_pcd;

pcl::search::KdTree<pcl::PointXYZ> _kdtreeLocalMap;
vector<int> _pointIdxRadiusSearch;
vector<float> _pointRadiusSquaredDistance;

void rcvGlobalPointCloudCallBack(const sensor_msgs::PointCloud2& pointcloud_map) {
  if (has_global_map) return;

  ROS_WARN("Global Pointcloud received..");

  pcl::PointCloud<pcl::PointXYZ> cloud_input;
  pcl::fromROSMsg(pointcloud_map, cloud_input);

  _voxel_sampler.setLeafSize(0.1f, 0.1f, 0.1f);
  _voxel_sampler.setInputCloud(cloud_input.makeShared());
  _voxel_sampler.filter(_cloud_all_map);

  _kdtreeLocalMap.setInputCloud(_cloud_all_map.makeShared());

  // 初始化occupancy map
  occupancy_buffer.resize(_GLX_SIZE * _GLY_SIZE * _GLZ_SIZE, 0);
  for (const auto& pt : _cloud_all_map.points) {
    Eigen::Vector3i idx = coord2gridIndex(Eigen::Vector3d(pt.x, pt.y, pt.z));
    size_t hash = idx(0) + _GLX_SIZE * (idx(1) + _GLY_SIZE * idx(2));
    occupancy_buffer[hash] = 1;
  }

  has_global_map = true;
}

void renderSensedPoints(const ros::TimerEvent &event)
{
  if (!has_global_map || !has_odom)
    return;

  Eigen::Vector3d sensor_pos(_odom.pose.pose.position.x,
                             _odom.pose.pose.position.y,
                             _odom.pose.pose.position.z);

  pcl::PointCloud<PointXYZIRT> _local_map;

  pcl::PointXYZ searchPoint(sensor_pos(0), sensor_pos(1), sensor_pos(2));
  _pointIdxRadiusSearch.clear();
  _pointRadiusSquaredDistance.clear();

  Eigen::Vector3d map_min, map_max;
  // 这里需要设置你的地图边界

  map_max << 21, 15, 2.5;    // 例如 (-50, -50, -5)
  map_min << -21, -15, -2.5; // 例如 (50, 50, 10)
  int occ_cnt = 0;
  for (auto v : occupancy_buffer)
    if (v > 0)
      occ_cnt++;

  int blocked_cnt = 0, visible_cnt = 0;

  if (_kdtreeLocalMap.radiusSearch(searchPoint, sensing_horizon,
                                   _pointIdxRadiusSearch,
                                   _pointRadiusSquaredDistance) > 0)
  {
    for (size_t i = 0; i < _pointIdxRadiusSearch.size(); ++i)
    {
      int idx_cloud = _pointIdxRadiusSearch[i];
      if (idx_cloud < 0 || idx_cloud >= _cloud_all_map.points.size())
      {
        ROS_ERROR_STREAM("Point index out of range: " << idx_cloud << " size=" << _cloud_all_map.points.size());
        continue;
      }
      pcl::PointXYZ base_pt = _cloud_all_map.points[idx_cloud];
      // pcl::PointXYZ base_pt = _cloud_all_map.points[_pointIdxRadiusSearch[i]];
      Eigen::Vector3d pt_vec(base_pt.x, base_pt.y, base_pt.z);

      // 射线采样
      std::vector<Eigen::Vector3d> ray_pts;
      Raycast(sensor_pos, pt_vec, map_min, map_max, &ray_pts);

      bool blocked = false;
      for (const auto &ray_pt : ray_pts)
      {
        // 判断ray_pt是否在障碍体素内

        Eigen::Vector3i idx = coord2gridIndex(ray_pt);
        size_t hash = idx(0) + _GLX_SIZE * (idx(1) + _GLY_SIZE * idx(2));
        if (hash >= occupancy_buffer.size())
        {
          ROS_ERROR_STREAM("occupancy_buffer out of range in map build! hash=" << hash << " size=" << occupancy_buffer.size());
          continue;
        }
        if (occupancy_buffer[hash] > 0)
        { // 有障碍
          blocked = true;
          // std::cout << "blocked :" << std::endl;
          break;
        }
      }

      if (blocked)
        blocked_cnt++;
      else
        visible_cnt++;
      if (blocked)
        continue; // 被遮挡则跳过

      PointXYZIRT pt;
      pt.x = base_pt.x;
      pt.y = base_pt.y;
      pt.z = base_pt.z;
      pt.intensity = 0;
      pt.ring = 0;
      pt.time = static_cast<float>(i) * 0.0001f;

      _local_map.points.push_back(pt);
    }
  }
  else
  {
    return;
  }
  _local_map.width = _local_map.points.size();
  _local_map.height = 1;
  _local_map.is_dense = true;

  sensor_msgs::PointCloud2 _local_map_pcd;
  pcl::toROSMsg(_local_map, _local_map_pcd);
  _local_map_pcd.header.frame_id = "map";
  _local_map_pcd.header.stamp = ros::Time::now();
  pub_cloud.publish(_local_map_pcd);
}

void rcvLocalPointCloudCallBack(
    const sensor_msgs::PointCloud2 &pointcloud_map)
{
  // do nothing, fix later
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pcl_render");
  ros::NodeHandle nh("~");

  nh.getParam("sensing_horizon", sensing_horizon);
  nh.getParam("sensing_rate", sensing_rate);
  nh.getParam("estimation_rate", estimation_rate);

  nh.getParam("map/x_size", _x_size);
  nh.getParam("map/y_size", _y_size);
  nh.getParam("map/z_size", _z_size);
  _gl_xl = -_x_size / 2.0;
  _gl_yl = -_y_size / 2.0;
  _gl_zl = 0.0;
  _resolution = 0.05;
  _inv_resolution = 1.0 / _resolution;

  _GLX_SIZE = (int)(_x_size * _inv_resolution);
  _GLY_SIZE = (int)(_y_size * _inv_resolution);
  _GLZ_SIZE = (int)(_z_size * _inv_resolution);

  // subscribe point cloud
  global_map_sub = nh.subscribe("global_map", 1, rcvGlobalPointCloudCallBack);
  local_map_sub = nh.subscribe("local_map", 1, rcvLocalPointCloudCallBack);
  odom_sub = nh.subscribe("odometry", 50, rcvOdometryCallbck);

  // publisher depth image and color image
  pub_cloud =
      nh.advertise<sensor_msgs::PointCloud2>("pcl_render_node/cloud", 10);

  pub_occupancy_buffer = nh.advertise<sensor_msgs::PointCloud2>("occupancy_buffer_pcl", 1); // 新增

  double sensing_duration = 1.0 / sensing_rate * 2.5;

  local_sensing_timer =
      nh.createTimer(ros::Duration(sensing_duration), renderSensedPoints);
  std::cout << "_x_size=" << _x_size << " _y_size=" << _y_size << " _z_size=" << _z_size << std::endl;
  std::cout << "_GLX_SIZE=" << _GLX_SIZE << " _GLY_SIZE=" << _GLY_SIZE << " _GLZ_SIZE=" << _GLZ_SIZE << std::endl;
  std::cout << "_resolution=" << _resolution << " _inv_resolution=" << _inv_resolution << std::endl;

  ros::Rate rate(100);
  bool status = ros::ok();
  while (status)
  {
    ros::spinOnce();
    status = ros::ok();
    rate.sleep();
  }
}
