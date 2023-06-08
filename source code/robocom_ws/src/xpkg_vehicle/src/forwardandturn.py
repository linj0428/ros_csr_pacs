#1 /usr/bin/env python
#-*- coding:utf-8 -*-

import rospy                                            #导入rospy
from geometry_msgs.msg import Twist  #导入必要数据类型
from math import pi                                #导入弧度制

linear_speed = 0.2
goal_distance = 5

angular_speed = 1.0
goal_angular = pi

def forwardandturn():
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 5)      #建立发布器，参数为主题命、数据类型、频率
    rospy.init_node('forward_and_turn', anonymous=True)         #初始化一个节点名字，参数为节点名字、随机数保证唯一
    rate = rospy.Rate(10)    #初始化一个Rate对象，通过后面的sleep()可以设定循环的频率

    for i in range(2):

        move_cmd = Twist()                          #定义数据结构
        move_cmd.linear.x = linear_speed    #赋值初始线性速度
        rospy.loginfo("linear.x:{}".format(move_cmd.linear.x))   #显示一个日志，在窗口，可以看见设定值
        for t in range(50):                                                           #循环50次
	    pub.publish(move_cmd)                                #发布线速度数据
	    rate.sleep()                                                  #循环频率

        move_cmd = Twist()
        rospy.sleep(1)
        move_cmd.angular.z = angular_speed           #设定赋值角速度
        rospy.loginfo("angular.z: {}".format(move_cmd.angular.z))
        for t in range(10):
             pub.publish(move_cmd)                                        #发布角速度
             rate.sleep()

if __name__ == "__main__":
    try:
        forwardandturn()                                            #运行控制小车前进和转向
    except rospy.ROSInterruptException:
        pass


