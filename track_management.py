#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:27:51 2019

@author: matthieu
"""

import numpy as np
import pickle
import os


def distance(pos1, pos2):
    return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))


class track:
    def __init__(self, color=None):
        self.poses = {}     # contains frame_id : position
        self.dims = {}      # contains frame_id : dimension
        if color is not None:
            self.color = color
        else:
            self.color = np.random.randint(0, 255, 3)


    def add_observation(self, frame_id, position, dim):
        if frame_id in self.poses.keys():
            print("Warning: a position is already known for this object at this frame.")
        self.poses[frame_id] = position
        self.dims[frame_id] = dim

    def last_observation(self):
        ids = sorted(self.poses.keys())
        max_idx = max(ids)
        return max_idx, self.poses[max_idx], self.dims[max_idx]

    def __str__(self):
        ids = sorted(self.poses.keys())
        s = ""
        for i in ids:
            s += str(i) + " " + str(self.poses[i]) + " " + str(self.dims[i]) + "   "
        return s

    def total_movement(self):
        ids = sorted(self.poses.keys())
        total = 0
        for i in range(1, len(ids)):
            p0 = self.poses[ids[i-1]]
            p1 = self.poses[ids[i]]
            total += distance(p0, p1)
        return total

class tracks_manager:
    def __init__(self, treshold_tracks_association=2):
        self.tracks = []
        self.THRESHOLD_TRACKS_ASSOCIATION = treshold_tracks_association
        pass

    def add_observation(self, frame_id, position, dim):
        BIG_NUMBER = 999999999
        dist_to_prev = [BIG_NUMBER] * len(self.tracks)
        for i, tr in enumerate(self.tracks):
            last_frame, last_pos, last_dim = tr.last_observation()
            if last_frame == frame_id - 1:      ## maybe or
                dist_to_prev[i] = distance(position, last_pos)
        dist_min = BIG_NUMBER
        dist_min_i = 0
        if len(dist_to_prev) > 0:
            dist_min_i = np.argmin(dist_to_prev)
            dist_min = dist_to_prev[dist_min_i]
        if dist_min > self.THRESHOLD_TRACKS_ASSOCIATION:
            self.tracks.append(track())  # add a new track
            dist_min_i = len(self.tracks) - 1
        self.tracks[dist_min_i].add_observation(frame_id, position, dim)

    def get_moving_tracks_idx(self):
        list_of_moving_tracks = []
        for i, tr in enumerate(self.tracks):
            if len(tr.poses.keys()) > 2 and tr.total_movement() < 2:
                continue
            list_of_moving_tracks.append(i)
        return list_of_moving_tracks
        

    def __str__(self):
        s = ""
        for i, track in enumerate(self.tracks):
            s += "track " + str(i) + " " + str(track) + "\n"
        return s


    
#def save_tracks_manager(tm, filename):
#    with open(filename, 'bw') as f:
#        pickle.dump(tm, f)
#        
#def load_tracks_manager(filename):
#    if os.path.exists(filename):
#        with open(filename, 'br') as f:
#            return pickle.load(f)
#    else:
#        return tracks_manager()
#    



def save(tm, filename):
    max_id = max([max(tr.poses.keys()) for tr in tm.tracks]) + 2
    # data layout:
    #   - each track is on two lines
    #   - first line for positions
    #   - second line for bb dims
    #   - the last col contains the colors
    data = np.zeros((len(tm.tracks) * 2, max_id + 1, 4))
    for i, tr in enumerate(tm.tracks):
        for k in sorted(tr.poses.keys()):
            data[2 * i, k, :3] = tr.poses[k]
            data[2 * i, k, 3] = 1
            data[2 * i + 1, k, :3] = tr.dims[k]
            data[2 * i + 1, k, 3] = 1
        data[2 * i, -1, :3] = tr.color
        data[2 * i + 1, -1, :3] = tr.color
    np.save(filename, data)
    
def load(filename):
    filename += ".npy"
    if os.path.exists(filename):
        data = np.load(filename)
        colors = data[::2, -1, :]
        positions = data[::2, :-1, :]
        dimensions = data[1::2, :-1, :]
        tm = tracks_manager()
        for i in range(positions.shape[0]):
            tr = track(colors[i, :3])
            for k in range(positions.shape[1]):
                if positions[i, k, 3] == 1:
                    tr.add_observation(k, positions[i, k, :3], dimensions[i, k, :3])
            tm.tracks.append(tr)
        return tm
    else:
        return tracks_manager()

tm = tracks_manager(1)
tm.add_observation(0, [1,1,1], [0.5, 0.5, 0.5])
tm.add_observation(0, [2,2,2,], [0.5, 0.5, 0.5])
print(tm)

tm.add_observation(1, [1.5,1,1],[0.5, 0.5, 0.5])
tm.add_observation(1, [2.5,1.9,1.8], [0.5, 0.5, 0.5])

print(tm)

tm.add_observation(2, [1.6,1,1], [0.5, 0.5, 0.5])
tm.add_observation(2, [2.6, 2, 2], [0.5, 0.5, 0.5])

print(tm)

tm.add_observation(3, [3, 2, 2], [0.5, 0.5, 0.5])

print(tm)

tm.add_observation(4, [1.7,1,1], [0.5, 0.5, 0.5])
tm.add_observation(4, [6, 6, 6], [0.5, 0.5, 0.5])

print(tm)

#save_tracks_manager(tm, "filesave.bin")
save(tm, "fsave.bin")
tm = []
print(tm)

#tm = load_tracks_manager("filesave.bin")
tm = load("fsave.bin")
print(tm)



print(tm.get_moving_tracks_idx())


