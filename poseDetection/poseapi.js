const express = require("express");
const bodyParser = require("body-parser");
const poseDetect = require("./poseDetect.js")

// express application
var app = express();

// add body-parser
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

//initialization
async function init() {
  //load model
  console.log("Loading network...")
  const net = await poseDetect.loadModel()
  console.log("Completed.")
  return net
}
const net = init()

// set routing.
app.use("/api/", (function () {
    var router = express.Router();

    router.post("/posedetect", (request, response) => {
        var body = request.body;
        var path = body.path;
        net.then(async function (result) {
          var pose = await poseDetect.getPose(result, path);
          response.json(pose);
        });
    });

    return router;
})());

// start web applicaiton.
app.listen(8000);
