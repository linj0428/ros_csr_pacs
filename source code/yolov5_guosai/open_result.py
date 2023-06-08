import cv2
import os
import time
time.sleep(100)
rootdir = "/home/eaibot/yolov5_guosai/renqun/images/result"
result_path='renqun/images/result/'
for filename in os.listdir(rootdir):
    img = cv2.imread(result_path + filename)
    img1 = cv2.resize(img, (996, 618))
    cv2.imshow("result", img1)
    cv2.waitKey(0)





