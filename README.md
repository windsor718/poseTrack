# PoseTrack - Bayesian Multi Camera racking and Pose Detection  
This code sets:  
- tracks person from single camera with Kalman filtering.  
- estimate poses of people on that image. Detect pre-defined actions.   
- associate and integrate one scene from multiple cameras with Bayesian filtering.  
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

JavaScript  
- Node.js  
- @tensorflow/tfjs-node  
- @tensorflow-models/posenet  
- canvas  
- express  
- body-parser

Use conda/pip and npm to install these libraries.  
## Notes  
Data association between cameras are done by estimating personal location on each image.  
This estimation was based on look-up table (image coordinates - real coordinates), created by 3D CG model in Autodesk 3dsmax.  
For general purposes, replace data association between cameras to metric-based algorithm.  

## Usage  
See README in each directory for further information on the components.   
### Running API server  
```javascript
node poseapi.js
```
The script firstly load a model from the google repositry. This will take minutes depending on a model size.  
In default the script uses **ResNet50** architecture with 801 input resolution and 16 output stride (maximum accuracy expected).  
After you load the model, the server is ready to use API!  
The server defines POST communication. Use POST to get the result from this server. The server listens localhost:8000 in default. Change this if needed.  
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
### future developments  
- implement deep metric association.
- efficient IO with SQLite
