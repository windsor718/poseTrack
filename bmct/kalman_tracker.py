"""
Original Repo: https://github.com/abewley/sort
Forked and edited by: Yuta Ishitsuka

Kalman Filtering approach for the single camera tracking.
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker(object):

    count = 0
    def __init__(self,bbox,id,img=None):
        """
        Kalman Filtering Tracking Class initializer

        Args:
            bbox (list): bounding box [x1,y1,x2,y2].
            id (int): initial id
            img (cv2.imageObject): opencv image
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0],
                            [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
                            [0,0,0,0,1,0,0], [0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0],
                            [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        #self.id = KalmanBoxTracker.count
        self.id = id
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox,img=None):
        """
        Updates the state vector with observed bbox.

        Args:
            bbox (list): bounding box [x1,y1,x2,y2].
            img (cv2.imageObject): opencv image

        Returns:
            None: internal update of instance variables.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if bbox != []:
            self.kf.update(convert_bbox_to_z(bbox))

    def predict(self,img=None):
        """
        Advances the state vec and returns the predicted bounding box estimate.

        Args:
            img (cv2.imageObject): opencv image

        Returns:
            list: most recent updated bounding box
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1][0]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)[0]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area
    and r is the aspect ratio.

    Args:
        bbox (list): bounding box [x1,y1,x2,y2].
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    #scale is just area
    r = w/float(h)
    return np.array([x, y, s, r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right

    Args:
        x (list): [x,y,s,r] where x,y is the centre of the box and s is
            the scale/area and r is the aspect ratio.
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if(score==None):
        return np.array([x[0]-w/2., x[1]-h/2.,
                            x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2.,
                            x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))
