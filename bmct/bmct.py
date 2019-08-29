"""
Coded  by: Yuta Ishitsuka

Bayesiam Multi Camera Tracking (BMCT)
Master class for the multi camera tracking system

Usage:
    1. create an instance. `ist = MultiCameraTracking()`
    1.5. (optional) edit ist variables for your specific case.
    2. register all cameras. `ist.registerCamera("camera001")`
    3. read pre-calculated image-real coordinates table and initialize dict.
        `ist.registerDataFrame()`
        `ist.initializeDict()`
    (calling ist.initialize(cameraNameList) will do 2. and 3. at once.)
    4. get integrated unique ids. `ist.multiCameraTracking(1)`
    5. repeat 4.

Notes:
    Edit getImage(**args) function for your purpose if it is applicable.

    Script-specific coding rule:
        ***s (plural form): list of ***
        ***List: list of ***

        * if the variable is the hierarchal list, the variable name is:
            ***sList, which is a list of list containing the groups of ***.
"""

import os
import cv2
import pandas as pd

from . import singleCameraTracking as sct
from . import integrateCameras as ic


class MultiCameraTracking(object):

    def __init__(self, useLocation=False):
        self.cameraDict = {}
        self.cameraList = []
        self.useLocation = useLocation
        if self.useLocation:
            sys.stderr.write("Location-based integration is beta version.")
        self.imageDir = os.path.join(os.getcwd(), "images/view")
        self.imageFmt = "%s_%d.jpg"
        self.dfDir = os.path.join(os.getcwd(), "3dsmax")
        self.priorsByCameras = {}
        self.uniqueIdsByCameras = {}
        self.uIdCount = 0
        self.threshold = 0.5

    def registerCamera(self, cameraName):
        """
        register new camera name in the instance.

        Args:
            cameraName (str): camera name
        """
        newInstance = sct.SingleCameraTracking()
        self.cameraList.append(cameraName)
        self.cameraDict[cameraName] = newInstance
        return 0

    def registerDataFrame(self, format="parquet"):
        """
        register data frames corresponding to the cameras in the instance.

        Args:
            format (str): default "parquet". Cached reference file from
                        3dsmax/getCoordinatesOnCamera.py. parquet or csv.
        """
        if format == "parquet":
            pathFormat = os.path.join(self.dfDir, "%s.parquet")
            self.calDfList = [pd.read_parquet(pathFormat % cameraName)
                              for cameraName in self.cameraList]
        elif format == "csv":
            pathFormat = os.path.join(self.dfDir, "%s.csv")
            self.calDfList = [pd.read_csv(pathFormat % cameraName, header=0)
                              for cameraName in self.cameraList]
        else:
            raise KeyError("data encoding format %s is not supported" % format)
        return 0

    def initializeDict(self):
        """
        initialize the prior probability dictionary (priorsByCameras) and
        unique ID disctionary over cameras (uniqueIdsByCameras).
        """
        for cameraName in self.cameraList:
            self.priorsByCameras[cameraName] = {}
            self.uniqueIdsByCameras[cameraName] = {}
        return 0

    def __checkConsistency(self):
        if len(self.cameraList) != len(self.calDfList):
            raise IOError("the lengthes of self.cameraList and self.dfList \
            must be same. Make sure to call registerDataFrame() \
            after you registered all cameras by registerCamera()")
        else:
            return 0

    def initialize(self, cameraNameList, format="parquet"):
        """
        initializing the instance for provided settings.

        Args:
            cameraNameList (list): list of camera names
            format (str): parquet or csv.
        """
        ret = [self.registerCamera(cameraName)
               for cameraName in cameraNameList]
        ret = self.initializeDict()
        if self.useLocation:
            ret = self.registerDataFrame(format=format)
            ret = self.__checkConsistency()
        return ret

    def readImageFromPath(self, imagePath):
        """
        read image from the specified path.

        Args:
            imagePath (str): image path

        Returns:
            cv2.imageObject
        """
        image = cv2.imread(imagePath)
        return image

    def getImage(self, cameraName, sceneNumber):
        """
        get one scene image from the specified camera.

        Args:
            cameraName (str): camera name
            sceneNumber (int): scene number

        Returns:
            cv2.imageObject
        """
        imageName = self.imageFmt % (cameraName, sceneNumber)
        imagePath = os.path.join(self.imageDir, imageName)
        image = self.readImageFromPath(imagePath)
        return image

    def singleCameraTracking(self, image, cameraName):
        """
        track the human object from a camera. IDs are unique in each camera,
        but not over multiple cameras.

        Args:
            image (cv2.imageObject): image object to track
            cameraName (str): corresponding camera name

        Returns:
            list: list of image points (bottom center X and Y) list.
            list: id list (camera-specific, not universal)

        Notes:
            BMCT-stand-alone version. Decided not to use with poseDetection.
            Use this func. only if you want to track people
            without pose Detection.
        """
        sctInstance = self.cameraDict[cameraName]
        trackers = sctInstance.getScene(image)
        ids = []
        imagePoints = []
        for idx, d in enumerate(trackers):
            bottomCenterX = (d[0] + d[2])/2.  # iamge coordinates
            bottomCenterY = d[3]  # image coordinates
            id = d[4]
            ids.append(id)
            imagePoints.append([bottomCenterX, bottomCenterY])
        return imagePoints, ids

    def __iterCameras(self, sceneNumber):
        idsList = []
        imagePointsList = []
        for cameraName in self.cameraList:
            image = self.getImage(cameraName, sceneNumber)
            imagePoints, ids = self.singleCameraTracking(image, cameraName)
            idsList.append(ids)
            imagePointsList.append(imagePoints)
        return imagePointsList, idsList

    def multiCameraTracking(self, imagePointsList, idsList):
        """
        Integrate the information from the singleCameraTracking and track the
        human objects using multiple cameras.

        Args:
            sceneNumber (int): scene number

        Retusns:
            dictionary: hierarchal dictionary of unique (universal) ids over
                multiple cameras. the dictionary has the structure like:
                {cameraName:camera-specific-id:uniqueId}
        """
        # imagePointsList, idsList = self.__iterCameras(sceneNumber)
        realPointsByCameras, idsByCameras = ic.CreateSceneByCameras(
                                    self.cameraList, imagePointsList,
                                    idsList, self.calDfList
                                    )
        self.uniqueIdsByCameras, count = ic.integrate(
                            self.cameraList, realPointsByCameras, idsByCameras,
                            self.priorsByCameras, self.uniqueIdsByCameras,
                            threshold=self.threshold, startCount=self.uIdCount
                            )
        self.uIdCount = count
        return self.uniqueIdsByCameras

###


def tester(format="parquet"):
    """
    For testing purpose

    Notes:
        Note that the resolution of the image-real world coordination table
        (created by codes under 3dsmax/) is not high enough because this is
        just a testing purpose.
        This will cause a slight degradation of the result.
    """
    cameraList = ["PhysCamera001", "PhysCamera002",
                  "PhysCamera003", "PhysCamera004"]
    inst = MultiCameraTracking()
    ret = inst.initialize(cameraList, format=format)
    uniqueIdsByCameras = inst.multiCameraTracking(0)
    print(uniqueIdsByCameras)
    return ret


if __name__ == "__main__":
    tester(format="csv")
