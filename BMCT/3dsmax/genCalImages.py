# -*- coding: utf-8 -*-
"""
coded by: Yuta Ishitsuka

Usage:
    1: open 3ds max
    2: hit F11
    3: Type: python.ExecuteFile($pathToThisFile)
"""

import MaxPlus

#setParameters
top = 100
bottom = 0
left = 0
right = 100
resolution = 10

width = 640
height = 480

def defineGrids():
    """define grid coordinates in a real world"""
    vertical_points = [bottom + i*resolution for i in range(0, int((top-bottom)/resolution))+1]
    horizontal_points = [left + i*resolution for i in range(0, int((right-left)/resolution))+1]
    return vertical_points, horizontal_points

def createObject(x,y):
    """Create sphere object in 3dsmax"""
    boxObject = MaxPlus.Factory.CreateGeomObject(MaxPlus.ClassIds.Sphere)
    boxObject.ParameterBlock.Radius.Value = resolution/4
    boxNode = MaxPlus.Factory.CreateNode(boxObject)
    boxNode.Position = MaxPlus.Point3(x,y,0)
    return boxNode

def render(x,y):
    """Render if the viewport is camera views"""
    render = MaxPlus.RenderSettings
    render.SetSaveFile(True)
    render.SetWidth(width)
    render.SetHeight(height)
    index = 0
    for view in MaxPlus.ViewportManager.Viewports:
        viewType = view.GetViewType()
        MaxPlus.ViewportManager.SetActiveViewport(index)
        if viewType == 8:
            camera = view.GetViewCamera()
            name = camera.GetName()
        else:
            continue
        outname = MaxPlus.PathManager.GetTempDir() \
            + r"\%s_%d_%d.jpg"%(name, x, y)
        render.SetOutputFile(outname)
        MaxPlus.RenderExecute.QuickRender()
        index = index + 1

def main():
    X, Y = defineGrids()
    for x in X:
        for y in Y:
            node = createObject(x, y)
            render(x, y)
            node.Delete()

if __name__ == "__main__":
    main()
