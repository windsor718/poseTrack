# PoseTrack - Bayesian Multi Camera Tracking and Pose Detection  
This code sets:  
- tracks person from single camera with Kalman filtering.  
- estimate poses of people on that image. Detect pre-defined actions.   
- re-identificate people in one scene using ResNet50 or other models supported.  

Each person on images will be associated with one global unique id over multiple cameras.  
The ids, actions, frame numbers, and other associated information are stored into a RDB with sqlite3.  

## Requirements
Python  
- opencv  
- numpy  
- pandas  
- json  
- filterpy 
- scikit-learn  
- Numba  
- pytorch

JavaScript  
- Node.js  
- @tensorflow/tfjs-node-gpu  
- @tensorflow-models/posenet  
- canvas  
- express  
- body-parser

Note that `tfjs-node-gpu` only supports CUDA compute capability>=6.0.  
Currently (Apr. 2020) `tfjs-node-gpu` supports CUDA=10.0, thus set up your environment with CUDA=10.0.  
One of the successful build was:  
- CUDA=10.0
- cuDNN=7
- pytorch=1.2
- torchvision=0.4.2
- node=12.16.2
- tensorflow-js=1.7.2
- posenet=2.2.1
  
Use conda/pip and npm to install these libraries.  
## Notes  
Data association between cameras can also be done by estimating personal location on each image.  
This estimation was based on calibrated look-up table (image coordinates - real coordinates), created by 3D CG model in Autodesk 3dsmax.  
See BMCT/3dsmax for further description. Using both ImageNet-based models and camera calibration may lead higher accuracy.  

## Usage  
See README in each directory for further information on the components.   
### Running API server  
```javascript
node poseapi.js
```
The script firstly load a model from the google repositry. This will take minutes depending on a model size.  
In default the script uses **ResNet50** architecture with 801 input resolution and 16 output stride (maximum accuracy expected).  
After you load the model, the server is ready to use API!  
The server defines POST communication. Use POST to get results from this server. The server listens localhost:8000 in default. Change this if needed.  
The request url is in default: http://localhost:8000/api/posedetect.
In python:   
```python
import requests
imgPath = "/path/to/the/image"
url = "http://localhost:8000/api/posedetect"
result = requests.post(url, data={"path": imgPath})
print(result.text)
```
### track person and pre-defined actions  
After setting up the server, this app is almost ready to go.  
```python
import poseTrack

# create tracker instance
tracker = poseTrack.PoseTrack()
# set your image file format associated with scene number.
tracker.imgPath = "./img/camera001_date_scene%05d.png"
# register camera names in your system.
tracker.cameraList = ["PhysCamera001", "PhysCamera002"]
# initialize system.
tracker.initialize()
# start tracking
for i in range(0, 10):
  result = tracker.poseTrack(i)
```
Remember to set up instance variables for your use, and initialize instance again by initialize() method.  
```
PoseTrack.poseTrack()
```
will return the dictionary of {CameraName:InCameraId:globalUniqueId}. globalUniqueId is unique over multiple cameras you registered.  
This implementation will be deprecated in future commit for the efficiency. Data will only be stored in the database, not in the instance.  
The result is stored in the SQLite database.   
