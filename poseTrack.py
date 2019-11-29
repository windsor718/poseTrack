import os
import numpy as np
import math
import cv2
import json
import sqlite3
import itertools
import glob
from multiprocessing import Pool
from bmct import bmct
from posedetection import poseClassificate as ps
from deepreid import identify

"""
Todo:
    multiprocessing
"""


class PoseTrack(object):

    def __init__(self):
        self.sceneBuffer = 10
        self.imgPathFmt = os.path.join(os.getcwd(), "images/")
        self.tracker = bmct.MultiCameraTracking()
        self.cameraList = ["camera001", "camera002"]
        self.dbName = "logs.db"
        self.columns = "ID INTEGER, uID STRING, action STRING, \
                        poseAngles STRING, date STRING, sceneNumber INTEGER, \
                        PRIMARY KEY (date, sceneNumber)"
        self.placeHolder = "(?, ?, ?, ?, ?, ?)"
        # ${cameraName}_${id}_s${sceneNumber}_${date}.jpg
        self.cacheImgFmt = "%s_%d_s%d_%s.jpg"
        self.cacheDir = "./cache/"
        self.reidentifier = identify.Reidentify()
        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir)
        self.cacheFormat = "parquet"  # only for location tracking
        con = sqlite3.connect(self.dbName)
        con.close()

    def initialize(self):
        self.tracker.initialize(self.cameraList, format=self.cacheFormat)
        for camera in self.cameraList:
            self.createTable(camera, self.columns)

    def createTable(self, tableName, tableContent):
        """
        create a new table in a database if not exists.

        Args:
            tableName (str): table name
            tableContent (str): column name and data type in SQL grammer

        Returns:
            None
        """
        con = sqlite3.connect(self.dbName)
        cursor = con.cursor()
        command = "CREATE TABLE IF NOT EXISTS %s(%s)"\
                  % (tableName, tableContent)
        cursor.execute(command)
        con.commit()
        con.close()

    def poseTrack(self, sceneNumber, date):
        #idsList, _ = self.__iterCameras(sceneNumber, date)
        idsList, _ = self.__iterCamerasMulti(sceneNumber, date, nProcess=4)
        self.getSceneFromMultipleCameras(sceneNumber, date)

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
        area1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        area2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        ix1 = max(rec1[0], rec2[0])
        iy1 = max(rec1[1], rec2[1])
        ix2 = min(rec1[2], rec2[2])
        iy2 = min(rec1[3], rec2[3])
        areai = (ix2 - ix1) * (iy2 - iy1)
        iou = areai / (area1 + area2 - areai)
        return iou

    def removeDoubleCount(self, boxes, threshold=0.85):
        """
        remove double counting of same person based on iou.
        """
        idxCombos = itertools.combinations(
                    np.arange(0, len(boxes)).tolist(), 2
                    )
        idxOut = []
        idxRemove = []
        for c in idxCombos:
            iou = self.calcIOU(np.array(boxes[c[0]]), np.array(boxes[c[1]]))
            if iou > threshold:
                idxOut.append(min(c))
                idxRemove.append(max(c))
            else:
                [idxOut.append(ci) for ci in c]
        #  dropduplicates
        idxOut = [x for x in list(set(idxOut)) if x not in idxRemove]
        return [boxes[idx] for idx in idxOut]

    def getSceneFromSingleCamera(self, imgPath, cameraName, sceneNumber, date):
        """
        get a tracked-scene from single camera using YOLO/Kalman-tracker.
        Then integrate it with pose estimations by poseNet.
        Association is done by IOUs.

        Args:
            img (cv2.imageObject): input image
            cameraName (str): unique camera name

        Returns:
            list: coordinates on a image (for location-based integration)
            list: id numbers (for location-based integration)
        """
        # bmct side
        sctInstance = self.tracker.cameraDict[cameraName]
        img = cv2.imread(imgPath)
        trackers = sctInstance.getScene(img)
        # poseNet side
        poses = ps.getPose(imgPath)
        boxes = self.removeDoubleCount(ps.getBoundingBoxes(poses))  # replace with nonMaximaSup?
        ids = []
        imagePoints = []
        for idx, d in enumerate(trackers):
            box_bmct = d[0:4]
            id = d[4]
            outPath = os.path.join(self.cacheDir,
                                   self.cacheImgFmt % (cameraName, id,
                                                       sceneNumber, date))
            self.writeImage(img, box_bmct, outPath)
            ious = []
            for box_pose in boxes:
                iou = self.calcIOU(box_bmct, box_pose)
                ious.append(iou)
            poseIdx = np.argmin(np.array(ious))
            sceneStart = max(0, sceneNumber - self.sceneBuffer)
            sceneTail = sceneNumber
            poseLog = self.getPoseLog(date, id, cameraName,
                                      sceneStart, sceneTail)
            poseLog = ps.getAnglesTimeSequence(poseLog,
                                               poses[poseIdx]["keypoints"])
            actions = ps.detect(poseLog, buffer=self.sceneBuffer)
            footPoint = ps.getFootPoint(box_bmct, poses[poseIdx]["keypoints"])
            if footPoint is not None:
                ids.append(id)
                imagePoints.append(footPoint)
            self.storeActionData(id, actions, cameraName,
                                 poseLog[-1], date, sceneNumber)
        return ids, imagePoints

    def storeActionData(self, id, actions, cameraName,
                        pose, date, sceneNumber):
        """
        store action data into database as JSON string format.

        Args:
            id (int): camera-dependent id
            actions (dict): action dictionary {action:bool}
            cameraName (str): cameraName (table name)
            date (str): date string
            sceneNumber (int): scene number

        Returns:
            None
        """
        con = sqlite3.connect(self.dbName)
        cursor = con.cursor()
        command = "REPLACE INTO %s VALUES %s" % (cameraName, self.placeHolder)
        actions_str = json.dumps(actions)
        pose_str = json.dumps(pose)
        holder = (id, "temp", actions_str, pose_str, date, sceneNumber)
        cursor.execute(command, holder)
        con.commit()
        con.close()

    def getPoseLog(self, date, id, tableName, start, tail):
        con = sqlite3.connect(self.dbName)
        cursor = con.cursor()
        command = "SELECT poseAngles FROM %s \
                   WHERE id == %d AND date == \"%s\" \
                   AND sceneNumber BETWEEN %d AND %d" \
                   % (tableName, id, date, start, tail)
        cursor.execute(command)
        result = cursor.fetchall()
        log = [json.loads(r[0]) for r in result]
        return log

    def __iterCameras(self, sceneNumber, date):
        idsList = []
        imagePointsList = []
        for cameraName in self.cameraList:
            path = self.getPath(sceneNumber, cameraName)
            imagePoints, ids = self.getSceneFromSingleCamera(path, cameraName,
                                                             sceneNumber, date)
            idsList.append(ids)
            imagePointsList.append(imagePoints)
        print(idsList, imagePointsList)
        return idsList, imagePointsList

    def __iterCamerasMulti(self, sceneNumber, date, nProcess=8):
        idsList = []
        imagePointsList = []
        argsList = [[sceneNumber, cameraName, date] for cameraName
                   in self.cameraList]
        with Pool(nProcess) as p:
            result = p.map(self.__eachCameraForMulti, argsList)
        result_t = zip(*result)
        return result_t[0], result_t[1]

    def __eachCameraForMulti(self, args):
        path = self.getPath(args[0], args[1])
        imagePoints, ids = self.getSceneFromSingleCamera(path, args[1],
                                                         args[0], args[2])
        return [ids, imagePoints]

    def getPath(self, sceneNumber, cameraName):
            imgName = "%s/%s_s%03d.jpg" % (cameraName, cameraName, sceneNumber)
            imgPath = os.path.join(self.imgPathFmt, imgName)
            return imgPath

    def writeImage(self, img, bbox, path):
        selImg = img[math.floor(bbox[1]):math.ceil(bbox[3]),
                     math.floor(bbox[0]):math.ceil(bbox[2])]
        cv2.imwrite(path, selImg)

    def getSceneFromMultipleCameras(self, sceneNumber, date):
        paths = glob.glob(os.path.join(self.cacheDir, "*_s%d_%s.jpg" % (sceneNumber, date)))
        ids = [path.split("/")[-1].split("_")[0]+"_"+path.split("_")[1] for path in paths]
        matchedIDs = self.reidentifier.reidentify(ids, paths)
        #print(matchedIDs)

if __name__ == "__main__":
    poseTracker = PoseTrack()
    poseTracker.initialize()
    date = "2019-8-18"
    for i in range(0, 20):
        poseTracker.poseTrack(i, date)
