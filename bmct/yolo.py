"""
Coded by: Yuta Ishitsuka

Human detection using Yolo.
"""
import os
import time
import numpy as np
import cv2

class YoloImage(object):

    def __init__(self):
        self.modelDir = os.path.join(os.getcwd(), "bmct/yolo")
        self.inpWidth = 416
        self.inpHeight = 416

        self.confThsld = 0.5
        self.nonMaximaSupThsld = 0.4

    def setVariables(self):
        """
        initialize variables from given variables. Edit if applicable.
        """
        labelPath = os.path.join(self.modelDir, "coco.names")
        with open(labelPath) as f:
            self.labels = f.read().strip().split("\n")
        np.random.seed(20)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3))
        weightpath = os.path.join(self.modelDir, "yolov3.weights")
        confpath = os.path.join(self.modelDir, "yolov3.cfg")
        self.net = cv2.dnn.readNetFromDarknet(confpath, weightpath)
        #self.net = cv2.dnn.readNetFromDarknet(weightpath, confpath)

    def predict(self, image):
        """
        detect human objects in the specified image.

        Args:
            image (cv2.imageObject): image object to track

        Returns:
            list: list of detected boxes which are lists of coordinates and ids.
            int: height of the input image
            int: width of the input image
        """
        (H, W) = image.shape[:2]
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.inpHeight, self.inpWidth), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)

        return layerOutputs, H, W

    def getBoundaryBox(self, layerOutputs, H, W):
        """
        get the boundary box from the yolo detection result.

        Args:
            layerOutputs (list): list of detected boxes
                                which are lists of coordinates and ids.
            H (float): image height
            W (float): image width

        Returns:
            list: list of boudning box
            classIds: list of ids
            confidences: list of confidence
        """
        boxes = []
        classIds = []
        confidences = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = float(scores[classId])
                if classId != 0:
                    continue
                if confidence > self.confThsld:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centX, centY, width, height) = box.astype(int)
                    llcrnrX = int(centX - (width/2))
                    llcrnrY = int(centY - (height/2))

                    boxes.append([llcrnrX, llcrnrY, int(width), int(height)])
                    classIds.append(classId)
                    confidences.append(confidence)
        return boxes, classIds, confidences

    def nonMaximaSupression(self, boxes, confidences):
        """
        supress the double counts based on the non maxima supression.

        Args:
            boxes (list): list of bounding boxes
            confidences (list): list of confidences

        Nots:
            the order of boxes and confidences must be same.
        """
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confThsld, self.nonMaximaSupThsld)
        return idxs

    def drawBox(self, image, boxes, idxs, classIds):
        """
        simple rectangler drawing function for the visualization.

        Args:
            image (cv2.imageObject): image object of a single scene
            boxes (list): list of bounding boxes
            idxs (list): list of index numbers of lists
            classIds (list): list of ids
        """
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in self.colors[classIds[i]]]
                cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)

        return image

    def yolo(self, image):
        """
        master wrapper of the class. Takes an image and returns the
        visualization results (cv2.imageObject) and list of bouding boxes.

        Args:
            image (cv2.imageObject): image of single scene.
        """
        layerOutputs, H, W = self.predict(image)
        boxes, classIds, confidences = self.getBoundaryBox(layerOutputs, H, W)
        idxs = self.nonMaximaSupression(boxes, confidences)
        supBoxes = [boxes[idx[0]] for idx in idxs]
        return self.drawBox(image, boxes, idxs, classIds), supBoxes
