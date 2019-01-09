#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:12:09 2018

@author: matthieu
"""

import numpy as np
from sklearn.cluster import DBSCAN

# load the grid
with open("accumulated_grid.obj", "r") as fin:
    lines = fin.readlines()
pts = []
for l in lines:
    if len(l) > 0 and l[0] == "v":
        pts.append(list(map(float, l[1:].split())))
pts = np.vstack(pts)

# cluster points with DBSCAN
clustering = DBSCAN(eps=0.75, min_samples=10).fit_predict(pts)

# create one color per instance
colors = np.random.randint(0, 255, (max(clustering) + 1, 3))
colors = np.vstack((colors, [0, 0, 0]))  # black at then end so that -1 correspond to it

# save grid clustered with colors
with open("clustered_grid.obj", "w") as fout:
    for i, p in enumerate(pts):
        fout.write("v " + " ".join(p.astype(str)) + " " +
                   " ".join(colors[clustering[i], :].astype(str)) + "\n")

# split points using their cluster index
clustered_pts = []
for cls in range(max(clustering) + 1):
    clustered_pts.append(pts[clustering == cls, :])



import matplotlib.pyplot as plt


# create a bounding box for each cluster
bounding_boxes = []
for points in clustered_pts:
    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)
    bb = np.array([[xmin, ymin, zmin],
                   [xmax, ymin, zmin],
                   [xmax, ymax, zmin],
                   [xmin, ymax, zmin],
                   [xmin, ymin, zmax],
                   [xmax, ymin, zmax],
                   [xmax, ymax, zmax],
                   [xmin, ymax, zmax]])
#    bounding_boxes.append(bb)

#    plt.figure()
#    plt.scatter(points[:, 0], points[:, 1])
#
#    
#    pts2d = points[:, :2]
#    c = pts2d.mean(axis=0)
#        
#    plt.ylim(c[1] - 3, c[1] + 3)
#    plt.xlim(c[0] - 3, c[0] + 3)
#    
#    pts2d -= c
#    res = np.linalg.eig(pts2d.T @ pts2d)
#
#    L = res[1][:, 0]
#    l = res[1][:, 1]
#    temp = [c]
#    for i in range(10):
#        temp.append(c + i * L)
#        temp.append(c - i * L)
#    temp = np.vstack(temp)
#    plt.scatter(temp[:, 0], temp[:, 1])
#    
#    temp = [c]
#    for i in range(10):
#        temp.append(c + i * l)
#        temp.append(c - i * l)
#    temp = np.vstack(temp)
#    plt.scatter(temp[:, 0], temp[:, 1])
#
#    proj_on_L = pts2d @ L
#    i_min = np.argmin(proj_on_L)
#    i_max = np.argmax(proj_on_L)
#    
#    proj_on_l = pts2d @ l
#    print(proj_on_l[i_min])
#    print(proj_on_l[i_max])
#    proj_on_l -= np.mean(proj_on_l[[i_min, i_max]])
#    nb_inf = len(np.where(proj_on_l < 0)[0])
#    nb_sup = len(np.where(proj_on_l > 0)[0])
#    if nb_inf > nb_sup:
#        i_mid = np.argmin(proj_on_l)
#    else:
#        i_mid = np.argmax(proj_on_l)
#    
#    pts2d += c
#    A = pts2d[i_min, :]
#    B = pts2d[i_max, :]
#    C = pts2d[i_mid, :]
#    
#    m = (A + B) / 2
#    D = m + (m - C)
#    
#    
#    bb2d = np.vstack((A, C, B, D))
#    plt.scatter(bb2d[:, 0], bb2d[:, 1])
#    
#    bb = np.vstack((A, C, B, D, A, C, B, D))
#    bb = np.hstack((bb, np.array([zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]).reshape((-1, 1))))


    bounding_boxes.append(bb)



# write bounding boxes
k = 1
bb_topology = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                        [3, 2, 6, 7], [1, 2, 6, 5], [3, 0, 4, 7]], dtype=int)
with open("bounding_boxes.obj", "w") as fout:
    for i, bb in enumerate(bounding_boxes):
        for pt in bb:
            fout.write("v " + " ".join(pt.astype(str)) + " " +
                       " ".join(colors[i, :].astype(str)) + "\n")
        for f in bb_topology + k:
            fout.write("f " + " ".join(f.astype(str)) + "\n")
        k += 8
