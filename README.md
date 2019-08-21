# PoseTrack - Multi Camera Bayesian Tracking and Pose Detection  
This code sets:  
- tracks person from single camera with Kalman filtering.  
- estimate poses of people on that image.  
- associate and integrate one scene from multiple cameras with Bayesian filtering.  
Each person on images will be assigned with one global unique id over multiple cameras.  
The ids, actions, and frame numbers are stored into a database with sqlite3.  

# Requirements
See README in each directory.  

## Notes  
Data association between cameras are done by estimating personal location on each image.  
This estimation was based on look-up table (image coordinates - real coordinates), created by 3D CG model in Autodesk 3dsmax (c).  
For general purposes, replace data association between cameras to metric-based algorithm.  
