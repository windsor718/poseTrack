import numpy as np
import requests
import time
import json
import cv2

"""
Todo:
    - calculate average confience score
    - filter score
    - cooperate with tracking system
"""

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
        dict: poses
    """
    result = requests.post(url, data={"path": imgPath})
    if result.status_code == 404:
        raise IOError("404 Error. Check if API server is running.")
    else:
        poses = json.loads(result.text)
    return poses


def detect(angles, buffer=10):
    """
    detect pre-defined actions. higher wrapper.

    Args:
        angles(list): temporal sequence of getAngles

    Returns:
        dict
    """
    if len(angles) < buffer:
        buffer = len(angles)

    armpitR = np.array([d["armpitAngleR"] for d in angles])[-1*buffer::]
    armpitL = np.array([d["armpitAngleL"] for d in angles])[-1*buffer::]
    elbowR = np.array([d["elbowAngleR"] for d in angles])[-1*buffer::]
    elbowL = np.array([d["elbowAngleL"] for d in angles])[-1*buffer::]

    actions = dict(reachingR=detectReaching(armpitR, elbowR),
                   reachingL=detectReaching(armpitL, elbowL),
                   disengagingR=detectDisengaging(armpitR, elbowR),
                   disengagingL=detectDisengaging(armpitL, elbowL))
    return actions


def getAnglesTimeSequence(log, keypoints):
    """
    get and append a rescent pose of a person.

    Args:
        log(list): list of dictionary of pose angles.
        keypoints(list): dictionary of rescent pose information

    Returns:
        list
    """
    rescentPose = getAngles(keypoints)
    log.append(rescentPose)
    print(log[-1])
    return log


def getAngles(keypoints):
    """
    detect actions of a person.

    Args:
        keypoints(dict): dictionary containing pose information

    Returns:
        dict: list of pre-defined angles
    """
    hipR = getCoordinates(keypoints, "rightHip")
    hipL = getCoordinates(keypoints, "leftHip")
    shoulderR = getCoordinates(keypoints, "rightShoulder")
    shoulderL = getCoordinates(keypoints, "leftShoulder")
    elbowR = getCoordinates(keypoints, "rightElbow")
    elbowL = getCoordinates(keypoints, "leftElbow")
    wristR = getCoordinates(keypoints, "rightWrist")
    wristL = getCoordinates(keypoints, "leftWrist")

    armpitR = calcArmpitAngle(hipR, shoulderR, elbowR)
    armpitL = calcArmpitAngle(hipL, shoulderL, elbowL)
    elbowAglR = calcElbowAngle(shoulderR, elbowR, wristR)
    elbowAglL = calcElbowAngle(shoulderL, elbowL, wristL)
    return {"armpitAngleR": armpitR, "armpitAngleL": armpitL,
            "elbowAngleR": elbowAglR, "elbowAngleL": elbowAglL}


def getCoordinates(keypoints, part):
    """
    get list of xy coordinate of the specified part

    Args:
        keypoints(dict): pose information
        part(str): pose name

    Returns:
        list: coordinate [x, y]
    """
    try:
        x = [d["position"]["x"] for d in keypoints if d["part"] == part][0]
        y = [d["position"]["y"] for d in keypoints if d["part"] == part][0]
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
    cos = (vec1*vec2).sum()/(np.sqrt((vec1**2).sum())*np.sqrt((vec2**2).sum()))
    angle = np.arccos(cos)*57.2958
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
    if armpitDiff == 1 and \
       elbowDiff == 1and \
       armpitAngles[-1] > armpitThsld and \
       elbowAngles[-1] > elbowThsld:
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
    if armpitDiff == -1 and \
       elbowDiff == -1 and \
       armpitAngles[-1] < armpitThsld and \
       elbowAngles[-1] < elbowThsld:
        result = True
    else:
        result = False
    return result


def getAveragedDifference(array, d=2, thsld=5):
    """
    split array and get difference between two chunks.

    Args:
        array(np.ndarray): input array
        d(int): default 2, an input array will be splitted into this number.
        thsld(int): default 5, an threshold value to say "increase/decrease"

    Returns:
        bool: True if increase False if decrease or equal
    """
    window = int(len(array)/d)
    diff = np.nanmean(array[-1*window::]) - np.nanmean(array[0:window])
    if thsld < diff:
        return 1
    elif -1*thsld < diff < thsld:
        return 0
    else:
        return -1


def test():
    """
    Tester
    """
    fileFmt = "/mnt/c/Users/winze/Downloads/poseNet/src/jpg/pickup_%03d.jpg"
    log = []
    for i in range(0, 150):
        file = fileFmt % i
        print(file)
        poses = getPose(file)
        log = getAnglesTimeSequence(log, poses[0]["keypoints"])
        actions = detect(log)
        time.sleep(1)
        img = cv2.imread(file.split(".")[0]+"_detected.png")
        offset = 100
        for action in actions:
            if actions[action]:
                print(action)
                cv2.putText(img, action, (20, offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                            (255, 255, 255), thickness=2)
                offset = offset + 100
        cv2.imwrite(file.split(".")[0]+"_labeled.png", img)
        print(actions)


test()
