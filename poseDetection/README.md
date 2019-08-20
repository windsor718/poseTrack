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
  

