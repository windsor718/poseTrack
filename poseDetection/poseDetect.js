const fs = require('fs')
const util = require('util')
const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet')
const Canvas = require('canvas'),
    Image = Canvas.Image,
    createCanvas = Canvas.createCanvas;

const architecture = 'ResNet50';
const outputStride = 16;
const inputResolution = 801;
const flipHorizontal = false;
const maxDetections = 5;
const scoreThsld = 0.7; //for detection
const confThsld = 0.8; //for drawing circle

exports.loadModel = async function loadModel() {
  // load the posenet model from a checkpoint
  const net = await posenet.load({
    architecture: architecture,
    outputStride: outputStride,
    inputResolution: inputResolution,
    quantBytes: 2
  });
  return net
}

async function estimatePoseOnImage(net, imageElement,
                                   flipHorizontal, outputStride) {
  //estimating poses on image
  const pose = await net.estimateMultiplePoses(imageElement, {
            flipHolizaontal: flipHorizontal,
            maxDetections: maxDetections,
            scoreThsld: scoreThsld}
            );
  return pose;
}

function drawPartsLine(ctx, keypoints, part1, part2) {
  var p1 =keypoints.filter(function(item, index){
    if (item.part == part1) return true;
  });
  var p2 = keypoints.filter(function(item, index){
    if (item.part == part2) return true;
  });
  p1 = p1[0].position;
  p2 = p2[0].position;
  var p1_x = p1.x,
    p1_y = p1.y,
    p2_x = p2.x,
    p2_y = p2.y;

  drawLine(ctx, p1_x, p1_y, p2_x, p2_y);
}

function drawPart(ctx, keypoints, part, r=10) {
  var p = keypoints.filter(function(item, index){
    if (item.part == part) return true;
  });

  score = p[0].score;
  p = p[0].position;
  drawCircle(ctx, p.x, p.y, r, score);
}

function drawLine(ctx, x1, y1, x2, y2) {
  // draw line between (x1, y1), (x2, y2) on ctx
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = "#00FFFF"
  ctx.stroke();
}

function drawCircle(ctx, x1, y1, r, score) {
  ctx.beginPath();
  ctx.arc(x1, y1, r, 0, 2*3.145);
  if (score > confThsld) {
    ctx.fillStyle = "#00FFFF";
  } else {
    ctx.fillStyle = "#CC0000";
}
  ctx.fill();
}

function drawBodyLine(ctx, pose) {
  drawPart(ctx, pose.keypoints, "nose", 5);
  drawPart(ctx, pose.keypoints, "rightEye", 5);
  drawPart(ctx, pose.keypoints, "leftEye", 5);
  drawPart(ctx, pose.keypoints, "rightEar", 5);
  drawPart(ctx, pose.keypoints, "leftEar", 5);
  drawPart(ctx, pose.keypoints, "rightShoulder");
  drawPart(ctx, pose.keypoints, "leftShoulder");
  drawPart(ctx, pose.keypoints, "rightElbow");
  drawPart(ctx, pose.keypoints, "leftElbow");
  drawPart(ctx, pose.keypoints, "rightWrist");
  drawPart(ctx, pose.keypoints, "leftWrist");
  drawPart(ctx, pose.keypoints, "rightHip");
  drawPart(ctx, pose.keypoints, "leftHip");
  drawPart(ctx, pose.keypoints, "rightKnee");
  drawPart(ctx, pose.keypoints, "leftKnee");
  drawPart(ctx, pose.keypoints, "rightAnkle");
  drawPart(ctx, pose.keypoints, "leftAnkle");
  drawPartsLine(ctx, pose.keypoints, "rightShoulder", "leftShoulder");
  drawPartsLine(ctx, pose.keypoints, "rightShoulder", "rightElbow");
  drawPartsLine(ctx, pose.keypoints, "leftShoulder", "leftElbow");
  drawPartsLine(ctx, pose.keypoints, "rightElbow", "rightWrist");
  drawPartsLine(ctx, pose.keypoints, "leftElbow", "leftWrist");
  drawPartsLine(ctx, pose.keypoints, "rightShoulder", "rightHip");
  drawPartsLine(ctx, pose.keypoints, "leftHip", "rightHip");
  drawPartsLine(ctx, pose.keypoints, "rightHip", "rightKnee");
  drawPartsLine(ctx, pose.keypoints, "rightKnee", "rightAnkle");
  drawPartsLine(ctx, pose.keypoints, "leftShoulder", "leftHip");
  drawPartsLine(ctx, pose.keypoints, "leftHip", "leftKnee");
  drawPartsLine(ctx, pose.keypoints, "leftKnee", "leftAnkle");
  drawPartsLine(ctx, pose.keypoints, "rightShoulder", "leftHip");
  drawPartsLine(ctx, pose.keypoints, "rightHip", "leftShoulder");
}

exports.getPose = async function getPose(net, path) {
  const img = new Image();
  img.src = path;
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const poses = await estimatePoseOnImage(net, canvas,
                                          flipHorizontal, outputStride);
  const pose = poses[0] // update later
  return pose.keypoints
  /*
  drawBodyLine(ctx, pose);

  var out = fs.createWriteStream('./Downloads/poseNet/out/'+fillIdx+'.png'),
    stream = canvas.pngStream();

  stream.on('data', function(chunk){
    out.write(chunk);
  });
  */
}
