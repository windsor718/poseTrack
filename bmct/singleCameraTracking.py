"""
Coded by: Yuta Ishitsuka

A wrapper for the single camera tracking using Appearance/KalmanFilter models.
"""
import cv2
from . import yolo
import numpy as np
from .sort import Sort

class SingleCameraTracking(object):

    def __init__(self):
        self.Yolo = yolo.YoloImage()
        self.Yolo.setVariables()
        self.tracker = Sort(use_dlib=False)

    def getScene(self, frame):
        """
        Track the object using kalman fileter.

        Args:
            frame (cv2.imageObject): image object of the single scene.

        Returns:
            list: list of tracked objects info. (list of coordinates and id)
        """
        detectedBoxes = self.detect(frame)
        trackers = self.kalmanFilter(detectedBoxes)
        return trackers

    def detect(self, frame):
        """
        Detect the human objects and create boundary boxes using Yolo.

        Args:
            frame (cv2.imageObject): image object of the single image.

        Returns:
            list: detected bounding boxes' coordinates.
        """
        out, boxes = self.Yolo.yolo(frame)
        cv2.imwrite("test.jpg", out)
        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        detectedBoxes = np.array([x1,y1,x2,y2]).T
        return detectedBoxes

    def kalmanFilter(self, detectedBoxes):
        """
        update the kalman filter instance.

        Args:
            detectedBoxes (list): detected bounding boxes to track from previous
                                scene.
        """
        trackers = self.tracker.update(detectedBoxes)
        return trackers
