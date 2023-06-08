
1: The robocom_ws, slam_ws, and YDLidar-SDK function packages in this directory are function packages provided by the organizing committee.

2: The yolov5_guosai folder includes our object detection module and result output module. Open the folder, where detect.py adds real-time detection.

Target screening, automatic removal of useless images and other functions of the detection module, the module can be run at the same time with the line patrol program, reducing the time required for crowd recognition tasks

open_result.py is a result output module that automatically displays the result after a certain period of time.

3: saveimage.py is an image acquisition module, which can subscribe to the topics published by the Obi camera and save the screenshot of the image to the specified folder at the set frequency.

4: tracking_final.py is a line patrol module, which realizes the function of setting different pid parameters and speed based on different error values, so as to complete the line patrol task accurately and efficiently.