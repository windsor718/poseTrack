# Pose Estimation Web API and pose classifier  
Running pose estimation web api server with nodejs/posenet, and process those for pose detection.  

## Requirements  
Node.js  
@tensorflow/tfjs-node  
@tensorflow-models/posenet  
canvas  
express  
body-parser  

## How to use  
### Running API server  
```javascript
node poseapi.js
```
The script firstly load a model from google repositry. This will take minutes depending on a model size.  
After you load the model, the server is ready to use API!  
The server defines POST communication. Use POST to get the result from this server.  
### get poses on a image  
```python
import poseClassificate as ps
ps.getPose(imagePath)
```
In python, call getPose function with string argument. This will return a json string of a image from given argument.  
Internally this code communicate with Node.js server to fetch the json. This pose estimation is done by poseNet by tensorflow.js.  
### detect actions by a temporal pose sequence.  
```python
# log is a list of temporal sequence of angles up to previous time step.
nlog = ps.getAnglesTimeSequence(log, pose[0]["keypoints"])  
actions = ps.detect(nlog)  
```
This part should vary with users. In default detection is based on the angles of joints.  
```python
ps.getAnglesTimeSequence()
```
gets joint angles from a pose and concatenate with log, which is a temporal sequence of angles up to previous time step.  
Based on this sequence (nlog), detection is done by rule-based algorithm.  
If you have sufficient amount of data, detection algo will be more flexible, such as SVM or other classifer.  
