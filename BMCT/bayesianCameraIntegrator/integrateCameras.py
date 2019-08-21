# -*- coding: utf-8 -*-
"""
Coded by: Yuta Ishitsuka

integrating the multiple cameras using the coordination transform and bayesian
update method.

Notes:
    Script specific coding rule:
        ***s (plural form): list
        ***List: list
        ***By??? : dictionary whose key is ??? and value is *** {???:***}.
"""
import itertools
import collections
import numpy as np
import json
from . import bayesianUpdate


def locatePointsInReal(imageX, imageY, df):
    """
    Locate image coordinates into the real coordinates.

    Args:
        imageX (float): x coordination in the image
        imageY (float): y coordination in the image
        df (pandas.DataFrame): coordination transfer table

    Returns:
        list: float x and y corrdination in the real world

    Notes:
        df should be indexed by imageX and imageY

    ToDo:
        make it fancier
    """
    imageX_grid = df["imageX"].values
    imageX_snapped = imageX_grid[np.argmin((imageX_grid - imageX)**2)]
    df_snapped = df[df.imageX == imageX_snapped]
    imageY_grid = df_snapped.imageY.values
    imageY_snapped = imageY_grid[np.argmin((imageY_grid - imageY)**2)]
    df_snapped = df_snapped[df.imageY == imageY_snapped]
    realX = df_snapped["realX"].values[0]
    realY = df_snapped["realY"].values[0]
    return [realX, realY]


def locateSceneInReal(imagePoints, df):
    """
    Locate points in a scene image onto the real coordinates.

    Args:
        imagePoints (list): list of point coordination list
        df (pandas.DataFrame): coordination transfer table

    Returns:
        list: points in tthe real coordinates

    Notes:
        just a wrapper of the locatePointsInReal(**args).
    """
    realPoints = [locatePointsInReal(point[0], point[1], df)
                  for point in imagePoints]
    return realPoints


def CreateSceneByCameras(cameraList, imagePointsList, idsList, dfList):
    """
    Create dictionaries of points (real coordinates) and camera-specific id.

    Args:
        cameraList (list): list of camera name
        imagePointsList (list): points (list) in an image of each camera
        idsList (list): list of id (list) in an image of each camera

    Returns:
        dict: {cameraName:realPoints(list)}
        dict: {cameraName:camera-specific-ids(list)}

    Notes:
        just a wrapper of the locateSceneInReal(**args).
        Note that points and ids are lists.
    """
    realPointsList = [locateSceneInReal(imagePoints, dfList[idx])
                      for idx, imagePoints in enumerate(imagePointsList)]
    realPointsByCameras = dict(zip(cameraList, realPointsList))

    idsByCameras = dict(zip(cameraList, idsList))
    return realPointsByCameras, idsByCameras


def integrate(cameraList, realPointsByCameras, idsByCameras, priorsByCameras,
              uniqueIdsByCameras, threshold=0.9, startCount=0):
    """
    Integrate camera-specific ids to unique integrated id.

    Args:
        cameraList (list): list of camera name
        idsByCameras (dict): {cameraName:camera-specific-ids(list)}
        threshold (float): threshold to consider as two objects are same
        startCount (int): counter for the unique integrated id

    Returns:
        dict: unique integrated id dictionary aligned with camera name.
              {cameraName:camera-specific-id:unique-integrated-id}
        int: counter for the unique integrated id

    Notes:
        uniqueIdsByCameras is a hierarchy dictionary.
        example: {camera1:
                    {0001:u0001,
                     0002:u0002,
                    },
                  camera2:
                    {0001:u0002,
                     0002:u0003}
                 }
    """
    cameraPairs = itertools.permutations(cameraList, 2)
    uniques = []
    for cameraPair in cameraPairs:
        priorsByCameras, uniques = classificate(cameraPair, idsByCameras,
                                                realPointsByCameras, uniques,
                                                priorsByCameras)
    uniqueIdsByCameras, count = reidentify(uniqueIdsByCameras, uniques,
                                           startCount=startCount)
    return uniqueIdsByCameras, count


def classificate(cameraPair, idsByCameras, realPointsByCameras,
                 uniques, priorsByCameras, threshold=0.9):
    """
    In a given camera pair,
    classify the people in two images by Bayesian update.

    Args:
        cameraPair (list): camera name pair
        idsByCameras (dict): {cameraName:camera-specific-ids(list)}
        realPointsByCameras (dict): {cameraname:realPoints(list)}
        uniques (list): list of grouped ids (list).

    Returns:
        dict: dictionary of prior probabilities
            {cameraName:camera-specific-id:prior probability}
        list: unique-integrated-ids

    Notes:
        priorsByCameras is a hierarchy dictionary.
        example: {camera1:
                    {0001:
                        {camera2:
                            {0001:0.001,
                             0002:0.999},
                         camera3:
                            {0001:0.001,
                             0002:0.002}
                        },
                     ***
                    },
                  ***
                 }
    """
    # base camera
    camera_base = cameraPair[0]
    ids_base = idsByCameras[camera_base]
    realPoints_base = realPointsByCameras[camera_base]
    # opponent camera; objects to be assigned
    camera_opponent = cameraPair[1]
    ids_opponent = idsByCameras[camera_opponent]
    realPoints_opponent = np.array(realPointsByCameras[camera_opponent]).T
    for idx, point in enumerate(realPoints_base):
        id_base = ids_base[idx]
        x_base = realPoints_base[idx][0]
        y_base = realPoints_base[idx][1]
        priorsByCameras, posteriorDict = update(
                    camera_base, id_base, x_base, y_base,
                    camera_opponent, ids_opponent, realPoints_opponent[0],
                    realPoints_opponent[1], priorsByCameras
                    )
        uniques = match(camera_base, id_base, camera_opponent,
                        posteriorDict, uniques, threshold=threshold)
    return priorsByCameras, uniques


def update(camera_base, id_base, x_base, y_base,
           camera_opponent, ids_opponent, xs_opponent, ys_opponent,
           priorsByCameras):
    """
    Update prior probabilities to posterior ones.

    Args:
        camera_base (str): base camera name (see Notes)
        id_base (str): a camera-specific-id from a image of camera_base
        x_base (float): a real x coordinate of id_base
        y_base (float): a real y coordinate of id_base
        camera_opponent (str): opponent camera name (see Notes)
        ids_oppnnent (str): multiple camera-specific-ids
                            from a image of camera_opponent
        xs_opponent (list): multiple x coordinates of id_oppnnent
        ys_opponent (list): multiple y coordinates of id_oppnnent
        priorsByCameras (dict): dictionary of prior probabilities
              {cameraName:prior probabilities of camera-specific-ids}

    Returns:
        dict: dictionary of prior probabilities
              {cameraName:prior probabilities of camera-specific-ids}
        dict: dictionary of posterior probabilities with camera-specific-ids
              {camera-specific-id:posteriorProbability}

    Notes:
        We select one id from camera_base.
        Then we calculate posterior probability
        using bayesian update for all ids in an image from camera_opponent.

    ToDo:
        Eliminate some of the for loop/ifelse statement.
    """
    distances_sq = (xs_opponent-x_base)**2 + \
                   (ys_opponent-y_base)**2
    distances = distances_sq**(0.5)
    if len(priorsByCameras) == 0:
        initProbs = bayesianUpdate.initialize(ids_opponent)
        updates = bayesianUpdate.bayesianUpdate(initProbs,
                                                distances)
    elif id_base in priorsByCameras[camera_base].keys() and \
                camera_opponent in priorsByCameras[camera_base][id_base].keys():
        data = np.array([v for v in
            priorsByCameras[camera_base][id_base][camera_opponent].values()])
        updates = bayesianUpdate.bayesianUpdate(data, distances)
    else:
        initProbs = bayesianUpdate.initialize(ids_opponent)
        updates = bayesianUpdate.bayesianUpdate(initProbs,
                                                distances)
    posteriorDict = dict(zip(ids_opponent, updates))
    # update prior probability by assigning posterior one for the next step
    #priorsByCameras[camera_base][id_base][camera_opponent] = posteriorDict
    if id_base not in priorsByCameras[camera_base].keys():
        priorsByCameras[camera_base][id_base] = {}
    priorsByCameras[camera_base][id_base][camera_opponent] = posteriorDict
    return priorsByCameras, posteriorDict


def match(camera_base, id_base, camera_opponent,
          posteriorDict, uniques, threshold=0.9):
    """
    Get ids from ids_oppnnent whose probability is more than thresold.
    Then create a group (list) of those matched ids.

    Args:
        camera_base (str): base camera name
        id_base (int): camera-specific-id in camera_base
        camera_opponent (str): opponent camera name
        posteriorDict (dict): {camera-specific-id(opponent):probability}
        uniques (list): list of groups (list) including ids matched.
        threshold (float): thresold of probability to consider as a match.

    Returns:
        list: updated uniques

    Notes:
        If there are multiple members exceeding threshold probability,
        the maximum-probability-id is chosen.
    """
    matchIds = [k for k, v in posteriorDict.items() if v > threshold]
    id_base_cai = "%s_%s" % (camera_base, id_base)  # cai: camera and id
    if len(matchIds) > 0:
        probs = [posteriorDict[key] for key in matchIds]
        oId_key = matchIds[np.argmax(probs)]  # most likely opponent id
        id_opponent_cai = "%s_%s " % (camera_opponent, oId_key)
        misscount = 0
        for group in uniques:
            if id_base_cai in list(set(group)):
                group.append(id_opponent_cai)
            else:
                misscount += 1
        if misscount == len(uniques):
            uniques.append([id_base_cai, id_opponent_cai])
    else:
        misscount = 0
        for group in uniques:
            if id_base_cai in list(set(group)):
                break
            else:
                misscount += 1
        if misscount == len(uniques):
            uniques.append([id_base_cai])
    return uniques


def reidentify(uniqueIdsByCameras, uniques, startCount=0):
    """
    Based on the uniques, assign camera-wide-unique-id.

    Args:
        uniqueIdsByCameras (dict): {cameraName:camera-specific-id:camera-wide-id}
        uniques (list): list of unique ids group (list)
        startCount (int): the number of id starts from in this function

    Returns:
        dict: updated uniqueIdsByCameras
        count: counter for the next time

    Notes:
        uniqueIdsByCameras is a hierarchy dictionary
        example:
            {camera001:{
                        001:u001,
                        002:u002
                        },
            camera002:{
                        001:u003,
                        002:u001
                        }
            }
    """
    count = startCount
    for group in uniques:
        nextUniqueId = "u%d" % count
        newUniqueId, count = getUniqueId(group, uniqueIdsByCameras,
                                         nextUniqueId, count)
        uniqueIdsByCameras = assignUniqueId(group, uniqueIdsByCameras,
                                            newUniqueId)
    return uniqueIdsByCameras, count


def getUniqueId(group, uniqueIdsByCameras, nextUniqueId, count):
    """
    Based on the previous classification, define whether getting new unique id
    or get id from previous classification.

    Args:
        group (list): list of matched ids
        nextUniqueId (str): unique-id assigned next
                            if no unique-id in previous step found.
        count (int): unique-id counter

    Returns:
        str: unique-id assigned to the input group
        int: next count
    """
    previousUniqueIds = []
    for id_cai in group:  # cai: camera and id
        cameraName, id = id_cai.split("_")
        if id in uniqueIdsByCameras[cameraName].keys():
            uniqueId = uniqueIdsByCameras[cameraName][id]
            previousUniqueIds.append(uniqueId)
        else:
            continue
    if len(previousUniqueIds) > 0:
        counter = collections.Counter(previousUniqueIds)
        uniqueId = counter.most_common()[0][0]
    else:
        uniqueId = nextUniqueId
        count = count + 1
    return uniqueId, count

def assignUniqueId(group, uniqueIdsByCameras, uniqueId):
    """
    update uniqueIdsByCameras based on the result of getUniqueId(**args).

    Args:
        group (list): list of matched ids
        uniqueIdsByCameras (dict): {cameraName:camera-specific-id:camera-wide-id}
        uniqueId (str): the camera-wide-unique-id assigned to the group

    Returns:
        dict: updated uniqueIdsByCameras
    """
    for id_cai in group:  # cai: camera and id
        cameraName, id = id_cai.split("_")
        uniqueIdsByCameras[cameraName][id] = uniqueId
        uniqueIdsByCameras[cameraName][id]
    return uniqueIdsByCameras

    def saveLogs(priorsByCameras, camera_base, mode="a"):
        """
        Simple logger to dump prior dictonary into json.

        Args:
            priorsByCameras (dict): dictionary of prior probabilities.
            camera_base (str): base camera name
            mode (str): "a" or "w".

        Returns:
            NoneType
        """
        with open("%s_%s.json", mode):
            json.dump(priorsByCameras[camera_base])
