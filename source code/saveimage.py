#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
rospy.sleep(21)

class save_image():
    def __init__(self):
        self.count = 0
        self.cvbridge = CvBridge()

    def message(self, data):
        print(data.encoding)

    def save_image(self, data):
        image = self.cvbridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image[28:1400]

        if self.count < 10:
            name = '00000{}'.format(self.count)
        elif self.count < 100 and self.count >= 10:
            name = '0000{}'.format(self.count)
        elif self.count < 1000 and self.count >= 100:
            name = '00{}'.format(self.count)
        else:
            name = '0000000000000'
	if self.count < 120:
        	cv.imwrite('/home/eaibot/yolov5_guosai/renqun/images/test_guosai/{}.jpg'.format(name), image)
		rospy.sleep(0.07)
        	print('image:  {}'.format(name))
        	self.count += 1
	else :
		rospy.sleep(100)
'''-------------define main----------------'''
if __name__ == '__main__':
    try:
        a = save_image()
        rospy.init_node('save_image', anonymous=True)
        rospy.Subscriber("camera/rgb/image_raw",
                         Image,
                         a.save_image)
	
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
