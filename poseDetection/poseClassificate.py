import numpy as np
import requests
import json


def loadLog(path):
    """
    load log.json file.

    Args:
        path(str): path to the log file formatted in json.

    Returns:
        dict
    """
    with open(path, "r") as f:
        log = json.load(f)
    return log


def getPose(imgPath, url="http://localhost:8000/api/posedetect"):
    """
    get pose information from poseDetectWebApi (api.js).

    Args:
        imgPath(str): path to an image

    Returns:
        dict: pose
    """
    result = requests.post(url, data={"path": "pickup_100.jpg"})
    if result.status_code == 404:
        raise IOError("404 Error. Check if API server is running.")
    else:
        pose = json.loads(result.text)
    return pose


def detect(angles, buffer=10):
    """
    detect pre-defined actions. higher wrapper.

    Args:
        angles(dict): temporal sequence of getAngles

    Returns:
        dict
    """
    if len(angles) < buffer:
        buffer = len(angles)

    armpitR = np.array([d["armpitR"] for d in angles])[-1*buffer::]
    armpitL = np.array([d["armpitL"] for d in angles])[-1*buffer::]
    elbowR = np.array([d["elbowR"] for d in angles])[-1*buffer::]
    elbowL = np.array([d["elbowL"] for d in angles])[-1*buffer::]

    actions = dict(reachingR=detectReaching(armpitR, elbowR),
                   reachingL=detectReaching(armpitL, elbowL),
                   disengagingR=detectDisengaging(armpitR, elbowR),
                   disengagingL=detectDisengaging(armpitL, elbowL))
    return actions


def getAnglesTimeSequence(log, poseJson):
    """
    get and append a rescent pose of a person.

    Args:
        log(list): list of dictionary of poses.
        poseJson(dict): dictionary of rescent pose information

    Returns:
        list
    """
    rescentPose = getAngles(poseJson)
    log.append(rescentPose)
    return log


def getAngles(poseJson):
    """
    detect actions of a person.

    Args:
        poseJson(dict): dictionary containing pose information

    Returns:
        dict: list of pre-defined angles
    """
    hipR = getCoordinates(poseJson, "rightHip")
    hipL = getCoordinates(poseJson, "leftHip")
    shoulderR = getCoordinates(poseJson, "rightShoulder")
    shoulderL = getCoordinates(poseJson, "leftShoulder")
    elbowR = getCoordinates(poseJson, "rightElbow")
    elbowL = getCoordinates(poseJson, "leftElbow")
    wristR = getCoordinates(poseJson, "rightWrist")
    wristL = getCoordinates(poseJson, "leftWrist")

    armpitR = calcArmpitAngle(hipR, shoulderR, elbowR)
    armpitL = calcArmpitAngle(hipL, shoulderL, elbowL)
    elbowAglR = calcElbowAngle(shoulderR, elbowR, wristR)
    elbowAglL = calcElbowAngle(shoulderL, elbowL, wristL)
    return {"armpitR": armpitR, "armpitL": armpitL,
            "elbowAngleR": elbowAglR, "elbowAngleL": elbowAglL}


def getCoordinates(poseJson, part):
    """
    get list of xy coordinate of the specified part

    Args:
        poseJson(dict): pose information
        part(str): pose name

    Returns:
        list: coordinate [x, y]
    """
    try:
        x = poseJson[part]["x"]
        y = poseJson[part]["y"]
    except(KeyError):
        x = np.nan
        y = np.nan
    return [x, y]


def calcArmpitAngle(hip, shoulder, elbow):
    """
    calculate armpit angle

    Args:
        hip(list): [x, y]
        shoulder(list): [x, y]
        elbow(list): [x, y]

    Returns:
        float: angle
    """
    angle = calcAngle(np.array(hip), np.array(shoulder), np.array(elbow))
    return angle


def calcElbowAngle(shoulder, elbow, wrist):
    """
    calculate elbow angle

    Args:
        shoulder(list): [x, y]
        elbow(list): [x, y]
        wrist(list): [x, y]

    Returns:
        float: angle
    """
    angle = calcAngle(np.array(shoulder), np.array(elbow), np.array(wrist))
    return angle


def calcAngle(p1, p2, p3):
    """
    calculate angle of p1p2p3.

    Args:
        p1(np.array): point coordiate 1.
        p2(np.array): point coordiate 2.
        p3(np.array): point coordiate 3.

    Returns:
        float: angle in degrees.
    """
    vec1 = p1 - p2
    vec2 = p3 - p2
    cos = (vec1*vec2).sum()/((vec1**2).sum()+(vec2**2).sum())
    angle = np.arccos(cos)
    return angle


def detectReaching(armpitAngles, elbowAngles, armpitThsld=30, elbowThsld=90):
    """
    Detect whether a person is reaching a shelf.

    Args:
        armpitAngles(np.array): array of rescent armpit angles
        elbowAngles(np.array): array of rescent elbow angles

    Returns:
        bool
    """
    armpitDiff = getAveragedDifference(armpitAngles)
    elbowDiff = getAveragedDifference(elbowAngles)
    if armpitDiff and \
       elbowDiff and \
       armpitAngles[-1] > armpitThsld and \
       elbowAngles > elbowThsld:
        result = True
    else:
        result = False
    return result


def detectDisengaging(armpitAngles, elbowAngles,
                      armpitThsld=30, elbowThsld=90):
    """
    Detect whether a person is reaching a shelf.

    Args:
        armpitAngles(np.array): array of rescent armpit angles
        elbowAngles(np.array): array of rescent elbow angles

    Returns:
        bool
    """
    armpitDiff = getAveragedDifference(armpitAngles)
    elbowDiff = getAveragedDifference(elbowAngles)
    if not armpitDiff and \
       not elbowDiff and \
       armpitAngles[-1] < armpitThsld and \
       elbowAngles < elbowThsld:
        result = True
    else:
        result = False
    return result


def getAveragedDifference(array, d=2):
    """
    split array and get difference between two chunks.

    Args:
        array(np.ndarray): input array
        d(int): default 2, an input array will be splitted into this number.

    Returns:
        bool: True if increase False if decrease or equal
    """
    window = len(array)/d
    diff = np.nanmean(array[-1*window::]) - np.nanmean(array[0:window])
    return True if diff > 0 else False
