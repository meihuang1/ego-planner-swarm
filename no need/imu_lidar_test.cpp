#include <Eigen/Dense>
#include <cmath>

using Eigen::Vector3d;
using Eigen::Quaterniond;

// 四元数乘法 a*b
inline Quaterniond quat_mult(const Quaterniond &a, const Quaterniond &b) {
    return Quaterniond(
        a.w()*b.w() - a.x()*b.x() - a.y()*b.y() - a.z()*b.z(),
        a.w()*b.x() + a.x()*b.w() + a.y()*b.z() - a.z()*b.y(),
        a.w()*b.y() - a.x()*b.z() + a.y()*b.w() + a.z()*b.x(),
        a.w()*b.z() + a.x()*b.y() - a.y()*b.x() + a.z()*b.w()
    );
}

// 由前后两个四元数求角速度（rad/s）
inline Vector3d quat_to_ang_vel(const Quaterniond &q_prev, const Quaterniond &q_curr, double dt) {
    Quaterniond q_prev_conj(q_prev.w(), -q_prev.x(), -q_prev.y(), -q_prev.z());
    Quaterniond dq = q_prev_conj * q_curr;

    // double w = std::clamp(dq.w(), -1.0, 1.0);
    double w = dq.w() > 1 ? 1 : dq.w();
    w = dq.w() < -1 ? -1 : dq.w();

    Eigen::Vector3d v(dq.x(), dq.y(), dq.z());
    double v_norm = v.norm();

    double phi = 2.0 * std::atan2(v_norm, w);
    Vector3d rot_vec;
    if (v_norm < 1e-8) {
        rot_vec = 2.0 * v;
    } else {
        Eigen::Vector3d axis = v / v_norm;
        rot_vec = axis * phi;
    }
    return rot_vec / dt;
}

// 梯形积分加速度 -> 速度
inline Vector3d integrate_lin_acc(const Vector3d &prev_vel, const Vector3d &lin_acc, const Vector3d &prev_lin_acc, double dt) {
    return prev_vel + 0.5 * (prev_lin_acc + lin_acc) * dt;
}

// 梯形积分速度 -> 位置
inline Vector3d integrate_vel(const Vector3d &prev_pos, const Vector3d &vel, double dt) {
    return prev_pos + vel * dt;
}

// 梯形积分角加速度 -> 角速度
inline Vector3d integrate_ang_acc(const Vector3d &prev_ang_vel, const Vector3d &ang_acc, const Vector3d &prev_ang_acc, double dt) {
    return prev_ang_vel + 0.5 * (prev_ang_acc + ang_acc) * dt;
}

// 角速度积分 -> 新四元数姿态
inline Quaterniond integrate_ang_vel(const Quaterniond &prev_quat, const Vector3d &ang_vel, double dt) {
    double omega_mag = ang_vel.norm();
    Quaterniond dq;
    if (omega_mag * dt < 1e-8) {
        dq = Quaterniond(1.0, 0.5*ang_vel.x()*dt, 0.5*ang_vel.y()*dt, 0.5*ang_vel.z()*dt);
    } else {
        double theta = omega_mag * dt;
        Eigen::Vector3d axis = ang_vel / omega_mag;
        dq = Quaterniond(std::cos(theta/2.0),
                         axis.x()*std::sin(theta/2.0),
                         axis.y()*std::sin(theta/2.0),
                         axis.z()*std::sin(theta/2.0));
    }
    Quaterniond quat_new = prev_quat * dq;
    quat_new.normalize();
    return quat_new;
}


#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

const double GRAVITY_Z = 9.80665;

Eigen::Vector3d pos_est(0,0,0);
Eigen::Vector3d vel_est(0,0,0);
Eigen::Vector3d ang_vel_est(0,0,0);
Eigen::Quaterniond ang_quat_est(1,0,0,0);

Eigen::Vector3d prev_lin_acc(0,0,0);
Eigen::Vector3d prev_ang_acc(0,0,0);
double last_t = -1;

ros::Publisher odom_pub;

void imu_callback(const sensor_msgs::Imu::ConstPtr &msg) {
    double t = msg->header.stamp.toSec();
    if (last_t > 0) {
        double dt = t - last_t;
        if (dt > 0) {
            // --- 获取线加速度并补偿重力 ---
            Eigen::Vector3d lin_acc(msg->linear_acceleration.x,
                                    msg->linear_acceleration.y,
                                    msg->linear_acceleration.z);
            lin_acc.z() -= GRAVITY_Z;

            // --- 获取角加速度 ---
            Eigen::Vector3d ang_acc(msg->angular_velocity.x,
                                    msg->angular_velocity.y,
                                    msg->angular_velocity.z);

            // --- 线速度积分 ---
            vel_est = integrate_lin_acc(vel_est, lin_acc, prev_lin_acc, dt);
            // --- 位置积分 ---
            pos_est = integrate_vel(pos_est, vel_est, dt);
            prev_lin_acc = lin_acc;

            // --- 角速度积分 ---
            ang_vel_est = integrate_ang_acc(ang_vel_est, ang_acc, prev_ang_acc, dt);
            prev_ang_acc = ang_acc;

            // --- 姿态积分 ---
            ang_quat_est = integrate_ang_vel(ang_quat_est, ang_vel_est, dt);

            // --- 发布 Odometry ---
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = msg->header.stamp;
            odom_msg.header.frame_id = "world";
            odom_msg.pose.pose.position.x = pos_est.x();
            odom_msg.pose.pose.position.y = pos_est.y();
            odom_msg.pose.pose.position.z = pos_est.z();
            odom_msg.pose.pose.orientation.w = ang_quat_est.w();
            odom_msg.pose.pose.orientation.x = ang_quat_est.x();
            odom_msg.pose.pose.orientation.y = ang_quat_est.y();
            odom_msg.pose.pose.orientation.z = ang_quat_est.z();
            odom_msg.twist.twist.linear.x = vel_est.x();
            odom_msg.twist.twist.linear.y = vel_est.y();
            odom_msg.twist.twist.linear.z = vel_est.z();
            odom_msg.twist.twist.angular.x = ang_vel_est.x();
            odom_msg.twist.twist.angular.y = ang_vel_est.y();
            odom_msg.twist.twist.angular.z = ang_vel_est.z();
            odom_pub.publish(odom_msg);
        }
    }
    last_t = t;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "imu_integrator");
    ros::NodeHandle nh;

    odom_pub = nh.advertise<nav_msgs::Odometry>("/odom_est", 10);
    ros::Subscriber imu_sub = nh.subscribe("/drone_0/imu", 10, imu_callback);

    ros::spin();
    return 0;
}
