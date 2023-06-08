#!/usr/bin/env python
# coding=utf-8

import rospy
from sensor_msgs.msg import Image
import cv2, cv_bridge
import numpy
from geometry_msgs.msg import Twist


class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.twist = Twist()

        # PID参数定义
        self.Kp = 0.045
        self.Ki = 0.0
        self.Kd = 0.06
        self.num = 0
        self.data = 0.0
        self.data1 = 0.0

        self.PIDOutput = 0.0  # PID控制器输出

        self.Error = 0.0
        self.LastError = 0.0
        self.LastLastError = 0.0

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = numpy.array([0, 0, 0])
        upper_black = numpy.array([90, 90, 90])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        masked = cv2.bitwise_and(image, image, mask=mask)

        # 在图像某处绘制一个指示，因为只考虑20行宽的图像，所以使用numpy切片将以外的空间区域清空
        h, w, d = image.shape
        search_top = 11 * h / 16
        search_bot = search_top + 30
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0
        # 计算mask图像的重心，即几何中心
        M = cv2.moments(mask)
        # print M
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)
            self.LastLastError = self.LastError
            self.LastError = self.Error
            self.Error = w / 2 - cx
            print
            self.Error
            if self.Error >= -150 and self.Error <= 150:
                self.Kp = 0.04
                self.Ki = 0.0
                self.Kd = 0.05
            else:
                self.Kp = 0.053
                self.Ki = 0.0
                self.Kd = 0.06
            # 计算增量
            IncrementalValue = self.Kp * (self.Error - self.LastError) + \
                               self.Ki * self.Error + self.Kd * (self.Error - 2 * self.LastError + self.LastLastError)
            # 计算输出
            self.PIDOutput += IncrementalValue
            if self.Error >= -30 and self.Error <= 30:
                self.twist.linear.x = 0.45
                self.twist.angular.z = float(self.PIDOutput) / 8
                self.cmd_vel_pub.publish(self.twist)
            else:
                self.twist.linear.x = 0.45
                self.twist.angular.z = float(self.PIDOutput) / 8
                # self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
        else:
            self.twist.linear.x = 0.45
            self.twist.angular.z = 0
            self.cmd_vel_pub.publish(self.twist)
            # rospy.Timer(rospy.Duration(3), self.stop,oneshot=True)
        # cv2.imshow("window", image)
        # cv2.imwrite('/eaibot/pic',image)
        cv2.waitKey(3)


rospy.init_node("opencv")
follower = Follower()
rospy.spin()
