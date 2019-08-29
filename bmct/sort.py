"""
Original Repo: https://github.com/abewley/sort
Forked and edited by: Yuta Ishitsuka
"""

from __future__ import print_function
import numpy as np
from .kalman_tracker import KalmanBoxTracker
#from correlation_tracker import CorrelationTracker
from .data_association import associate_detections_to_trackers


class Sort:

  def __init__(self, max_age=1, min_hits=3, use_dlib=False):
    """
    Args:
        max_age (int): how many updates the unassigned id bare with?
        min_hits (int): minimum threshold of the hit_streak
        use_dlib (bool): True for appearance model, False for the Kalman Filter
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.idCount = 0

    self.use_dlib = use_dlib

  def update(self,dets,img=None):
    """
    Update states of the SORT instance.

    Args:
        dets (np.ndarray): an array of detections in the format
            [[x,y,w,h,score],[x,y,w,h,score],...]

    Returns:
        np.ndarray: [x,y,w,h,id]

    Notes:
        this method must be called once for each frame even with empty detections.
        The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict(img) #for kal!
      #print(pos)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    if dets != []:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

      #update matched trackers with assigned detections
      for t,trk in enumerate(self.trackers):
        if(t not in unmatched_trks):
          d = matched[np.where(matched[:,1]==t)[0],0]
          trk.update(dets[d,:][0],img) ## for dlib re-intialize the trackers ?!

      #create and initialise new trackers for unmatched detections
      for i in unmatched_dets:
        if not self.use_dlib:
          trk = KalmanBoxTracker(dets[i,:], self.idCount)
          self.idCount += 1
        else:
          trk = CorrelationTracker(dets[i,:],img)
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        if dets == []:
          trk.update([],img)
        d = trk.get_state()
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
