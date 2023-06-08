#include <ros/ros.h>
#include <ros/console.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
nav_msgs::Path  path_com;
ros::Publisher  path_pub_com;
 
void odomCallback(const nav_msgs::Odometry::ConstPtr& odom_com)
{
    geometry_msgs::PoseStamped this_pose_stamped_com;
    this_pose_stamped_com.pose.position.x = odom_com -> pose.pose.position.x;
    this_pose_stamped_com.pose.position.y = odom_com -> pose.pose.position.y;
    this_pose_stamped_com.pose.position.z = odom_com -> pose.pose.position.z;
 
    this_pose_stamped_com.pose.orientation = odom_com -> pose.pose.orientation;
 
    this_pose_stamped_com.header.stamp = ros::Time::now();
    this_pose_stamped_com.header.frame_id = "map";
 
    path_com.poses.push_back(this_pose_stamped_com);
 
    path_com.header.stamp = ros::Time::now();
    path_com.header.frame_id = "map";
    path_pub_com.publish(path_com);
    printf("path_pub:");
    printf("odom %.3lf %.3lf %.3lf\n", odom_com->pose.pose.position.x, odom_com->pose.pose.position.y, odom_com->pose.pose.position.z);
}
 
int main (int argc, char **argv)
{
    ros::init (argc, argv, "showpath_odom_com");
    ros::NodeHandle ph;
 
    path_pub_com = ph.advertise<nav_msgs::Path>("trajectory_odom_com", 10, true);
    ros::Subscriber odomSub = ph.subscribe<nav_msgs::Odometry>("/odom", 10, odomCallback);  //订阅里程计话题信息
    
    ros::Rate loop_rate(50);
    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}