"""
coded by: Yuta Ishitsuka
"""
import os
import glob

import cv2
import numpy as np
import pandas as pd

def getImages(dirPath):
    return glob.glob(dirPath+"/*")

def getCenter(imagePath):
    """Get the center of the object from the grayscale image
    Args:
        imagePath (str): path to the image

    Returns:
        x,y (float): center of the detected object
    """
    img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
    mu = cv2.moments(img, False)
    try:
        x, y = int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
    except(ZeroDivisionError):
        x, y = np.nan, np.nan
    return x, y

def getRelations(imagePath):
    """Get the relationship between the real and image coordinates
    Args:
        imagePath (str): path to the image

    Returns:
        relation (list): list of coordinates (realX, imageX, realY, imageY)
    """
    imageName = imagePath.split("/")[-1].split(".")[0]
    realX, realY = imageName.split("_")[1:3]
    imageX, imageY = getCenter(imagePath)
    relation = [realX, imageX, realY, imageY]
    return relation

def getCoordinates(dirPath):
    """iteration through the directory"""
    images = getImages(dirPath)
    emptyCheck(images)
    relations = [getRelations(path) for path in images]
    return relations

def emptyCheck(iList):
    if len(iList) == 0:
        raise IOError("No file found. Check your path to the image files.")

def convertToDataFrame(relations):
    """convert list to pandas.DataFrame with multiIndex"""
    df = pd.DataFrame(relations)
    df.columns = ["realX","imageX","realY","imageY"]
    df = df.dropna(how="any")
    df = df.set_index(["imageX","imageY"])
    return df

def main(dirPath):
    """main wrapper"""
    relations = getCoordinates(dirPath)
    df = convertToDataFrame(relations)
    return df

if __name__ == "__main__":
    cameraList = ["PhysCamera001","PhysCamera002","PhysCamera003","PhysCamera004"]
    imgRoot = "../../images/calImages_res500"
    for camera in cameraList:
        dirPath = imgRoot
        df = main(dirPath)
        #df.to_parquet("%s.parquet"%(camera))
        df.to_csv("%s.csv"%(camera))
