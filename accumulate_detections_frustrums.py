#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 08:44:40 2018

@author: matthieu
"""

import numpy as np
import glob
import os
from detect_frustrums import load_detections_2d, assign_labels, T_lidar_to_cam, P_cam, get_rotation_around_axis


# transforms generated by the SLAM
transforms_file = "/home/matthieu/Dev/VeloView-kwinternal/transforms_39.csv"


# load transforms (3x4)
with open(transforms_file, "r") as fin:
    lines = fin.readlines()
transforms = [np.eye(4)[:3, :]]
for i in range(1, len(lines)):      # skip the first line
    if len(lines[i]) > 0:
        temp = list(map(float, lines[i].split(",")))
        t = temp[0]
        roll = temp[1]
        pitch = temp[2]
        yaw = temp[3]
        x = temp[-3]
        y = temp[-2]
        z = temp[-1]
    rx = get_rotation_around_axis(roll, "x")
    ry = get_rotation_around_axis(pitch, "y")
    rz = get_rotation_around_axis(yaw, "z")
    R = rz @ ry @ rx
    T = np.hstack((R, np.array([[x, y, z]]).reshape((3, 1))))
    transforms.append(T)


labels_to_keep = ["none", "car", "person"]
labels_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]


detections_filename = "/home/matthieu/Lib/kwiver/build-sprokit/examples/pipelines/darknet/output/images_39/detections.csv"


pc_list = glob.glob("/media/matthieu/DATA/KITTI/2011_09_26_drive_0039_sync/velodyne_points/data/*.bin")
pc_list = sorted(pc_list)
pc_names = []
for f in pc_list:
    pc_names.append(os.path.splitext(os.path.basename(f))[0])


voxel_grid_scale = 10       # 1 means meters, 10 means cm, ...
grid = {}
total_pts = []
total_labels = []
for i, pc_file in enumerate(pc_list[:-1]):
    print("pointcloud ", i)
    # load point cloud
    pts = np.fromfile(pc_file, np.float32, -1).reshape((-1, 4))
    pts = pts[:, :3]

    # load detections
    detections_2d = load_detections_2d(detections_filename, pc_names[i])
    # assign labels
    labels = assign_labels(pts, T_lidar_to_cam, P_cam, 1242,
                           375, detections_2d, labels_to_keep)
    # transform points
    pts = transforms[i][:, :3] @ pts.T + transforms[i][:, 3].reshape((3, 1))
    pts = pts.T

    car_index = np.where(labels == 1)[0]
    detected_points = pts[car_index, :]
    detected_points = np.round(detected_points * voxel_grid_scale).astype(int)
    for p in detected_points:
        s = "|".join(p.astype(str))
        if s in grid.keys():
            grid[s] += 1
        else:
            grid[s] = 1

    total_pts.append(pts)
    total_labels.append(labels)
print("Writing pointcloud...")


with open("accumulated_grid.obj", "w") as fout:
    for key in grid:
        pt = list(map(lambda x: float(x)/voxel_grid_scale, key.split("|")))
        if grid[key] > 10:
            fout.write("v " + " ".join(map(str, pt)) + "\n")

# =============================================================================
# Write one pointcloud per frame transformed
# =============================================================================
#for i, pc in enumerate(total_pts):
#    with open("registered_pointclouds/pc_"+str(i)+".obj", "w") as fout:
#        for j, p in enumerate(pc):
#            fout.write("v " + " ".join(p.astype(str)) +
#                       " " + " ".join(map(str, labels_colors[total_labels[i][j]])) + "\n")


# =============================================================================
#  Write pointcloud containing all frames registered
# =============================================================================
total_pts = np.vstack(total_pts)
total_labels = np.concatenate(total_labels)
with open("registered_pointcloud_35.obj", "w") as fout:
    for i, p in enumerate(total_pts[::3, :]):
        fout.write("v " + " ".join(p.astype(str)) + " " + " ".join(map(str, labels_colors[total_labels[i]])) + "\n")
