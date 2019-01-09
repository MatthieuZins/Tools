#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:50:10 2019

@author: matthieu
"""

import numpy as np
import glob
import os



pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_26_drive_0017_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/2011_09_28_drive_0034_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/03/velodyne/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_29_drive_0026_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_28_drive_0034_sync/velodyne_points/data/*.bin")
#pc_list = glob.glob("/media/matthieu/DATA/KITTI/fixed/2011_09_28_drive_0053_sync/velodyne_points/data/*.bin")


pc_list = sorted(pc_list)
pc_names = []
for f in pc_list:
    pc_names.append(os.path.splitext(os.path.basename(f))[0])

if not os.path.isdir("ply"):
    os.mkdir("ply")


for i, pc_file in enumerate(pc_list[:-1]):
    print("pointcloud ", i)
    # load point cloud
    pts = np.fromfile(pc_file, np.float32, -1).reshape((-1, 4))
    pts = pts[:, :3]

    # Write pointcloud with a color per class
    header = "ply\n" \
             "format ascii 1.0\n" \
             "comment VCGLIB generated\n" \
             "element vertex " + str(pts.shape[0]) + "\n" \
             "property float x\n" \
             "property float y\n" \
             "property float z\n" \
             "element face 0\n" \
             "property list uchar int vertex_indices\n" \
             "end_header\n"
    out_dir_ply = "ply/"
    with open("ply/" + "pc_" + pc_names[i] + ".ply", "w") as fout:
        fout.write(header)
        for i, p in enumerate(pts):
            fout.write(" ".join(p.astype(str)) + "\n")
