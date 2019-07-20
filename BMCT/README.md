# Pass: Multi Camera Tracking System using Bayesian filtering.  
Camera calibration with the 3dsmax and a multi camera intregration with Bayesian filtering.  
This module sets are based on Python 3.6 or later.    
**Note that the sample datasets' resolution is very coarse (due to the limitation of data size in repo.), so one should allow the inaccuracy of the results in some extent. The real operation will be conducted with way higher resolution, so this inaccuracy will not happen.**
## 3dsmax  
*Note that one needs to create an own 3DCG model to represent real world coordinates.*  
Once your 3dsmax 3DCG model is created, set up the physical camera in accordance to your setting.  
Then use `genCalImages.py` in the 3dsmax MaxScript console (F11). Note that MaxScript only supports the Python 2.7, so only this script is based on Python2. See the code for further description.  
After the execution, use getCoordinatesOnCamera.py to get the relationship between real world coordinates and image coordinates on each camera.  
The results will be saved as parquet format.  
## bayesianCameraIntegrator  
Using Bayesian updates and filtering, `integrateCameras.py` redundantly (over time) categorizes (groups) the people shown in each camera.  
Note that we never know how many people are actually shown in the camera, so this is a free-n clustering problem.  
Given one object in specific camera, by assuming the distance to objects in other cameras represents a likelihood one can estimate the posterior probability to assign the one to objects in other cameras.  
`integrateCameras.py` is just a wrapper for this project, and actual Bayesian updates and filtering are given by `bayesianUpdates.py`.  
## singleCamera
**You need to download coco.names, yolov3.weights, and yolov3.cfg from https://pjreddie.com/darknet/yolo/ and place it under singleCamera/yolo/.**  
Contains the code sets of single camera tracking, which gives the tracking information of each camera. The human detections are done by Yolo_v3. The tracking part is currently done by Kalman filtering, which has a competitive tracking redundancy in an occlusion-frequent situation. The sample code of the appearance model is also included, but it is not integrated still. Note that part of the work is referenced and edited from the open-source work from Oxford (https://github.com/ZidanMusk/experimenting-with-sort).  
