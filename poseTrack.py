import numpy as np
import cv2
from BMCT import bmct
from poseDetection import poseclassificate as ps

"""
Todo:
    create database with sqlite3
    implement database-related functions
      - storeIdData()
      - storeActionData()
"""


class PoseTrack(object):

    def __init__(self):
        self.imgPathFmt = "./images/"
        self.tracker = bmct.MultiCameraTracking()
        self.cameraList = ["PhysCamera001", "PhysCamera002"]

        self.cacheFormat = "parquet"

    def initialize(self):
        self.tracker.initialize(self.cameraList, format=self.cacheFormat)

    def poseTrack(self, sceneNumber):
        imagePointsList, idsList = self.__iterCameras(sceneNumber)
        uniqueIdsByCameras = self.getSceneFromMultipleCameras(imagePointsList,
                                                              idsList)
        self.storeIdData(uniqueIdsByCameras)
        return uniqueIdsByCameras

    def calcIOU(self, rec1, rec2):
        """
        calculate IOU composed of rec1 and rec2.

        Args:
            rec1 (np.array): Rectangle 1 shaped (4,). See below.
            rec2 (np.array): Rectangle 2 shaped (4,).

        Returns:
            float: IOU

        Notes:
            format of input array: [x1, y1, x2, y2]
            (x1, y1)---------|
            |                |
            |---------(x2, y2)
        """
        area1 = (rec1[2] - rec1[0]) * (rec1[1] - rec1[3])
        area2 = (rec2[2] - rec2[0]) * (rec2[1] - rec2[3])
        ix1 = np.max(rec1[0], rec2[0])
        iy1 = np.max(rec1[1], rec2[1])
        ix2 = np.min(rec1[2], rec2[2])
        iy2 = np.min(rec1[3], rec2[3])
        areai = (ix2 - ix1) * (iy2 - iy1)
        iou = areai / (area1 + area2)
        return iou

    def getSceneFromSingleCamera(self, imgPath, cameraName):
        """
        get a tracked-scene from single camera using YOLO/Kalman-tracker.
        Then integrate it with pose estimations by poseNet.
        Association is done by IOUs.

        Args:
            img (cv2.imageObject): input image
            cameraName (str): unique camera name

        Returns:
            list: coordinates on a image
            list: id numbers
        """
        # bmct side
        sctInstance = self.cameraDict[cameraName]
        img = cv2.imread(imgPath)
        trackers = sctInstance.getScene(img)
        # poseNet side
        poses = ps.getPose(imgPath)
        boxes = ps.getBoundingBoxes(poses)

        ids = []
        imagePoints = []
        for idx, d in enumerate(trackers):
            box_bmct = d[0:4]
            id = d[4]
            ious = []
            for box_pose in boxes:
                iou = self.calcIOU(box_bmct, box_pose)
                ious.append(iou)
            poseIdx = np.argmin(np.array(ious))
            poseLog = self.getPoselog(id, cameraName)
            poseLog = ps.getAnglesTimeSequence(poseLog,
                                               poses[poseIdx]["keypoints"])
            actions = ps.detect(poseLog)
            ids.append(id)
            imagePoints.append(ps.getFootPoint(poses[poseIdx]["keypoints"]))
            self.storeActionData(actions)
        return ids, imagePoints

    def __iterCameras(self, sceneNumber):
        idsList = []
        imagePointsList = []
        for cameraName in self.cameraList:
            path = self.getPath(sceneNumber, cameraName)
            imagePoints, ids = self.getSceneFromSingleCamera(path, cameraName)
            idsList.append(ids)
            imagePointsList.append(imagePoints)
        return imagePointsList, idsList

    def getSceneFromMultipleCameras(self, imagePointsList, idsList):
        uniqueIdsByCameras = self.tracker.multiCameraTracking(imagePointsList,
                                                              idsList)
        return uniqueIdsByCameras
