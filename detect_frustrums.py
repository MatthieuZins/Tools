#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:26:56 2018

@author: matthieu
"""

import numpy as np
import cv2
import os
import glob


def load_detections_2d(filename, image_name):
    """ Load the detections found for an image. It creates a dict
        with labels as keys and a list of bounding boxes as values """
    detections_2d = {}
    with open(filename, "r") as fin:
        lines = fin.readlines()
    for l in lines:
        if len(l) > 0 and l[0] != "#":
            values = l.split(",")
            frame = values[1]
            if os.path.splitext(os.path.basename(frame))[0] == image_name:
                x0 = int(values[2])
                y0 = int(values[3])
                x1 = int(values[4])
                y1 = int(values[5])
                cls = values[7]
                prob = float(values[6])
                if cls in detections_2d.keys():
                    detections_2d[cls].append((x0, y0, x1, y1))
                else:
                    detections_2d[cls] = [(x0, y0, x1, y1)]
    return detections_2d


def assign_labels(pts, T_lidar_to_cam, P_cam, w, h, detections_2d, labels_to_keep):
    """ assign label to each 3D points using the 2D detections provided.
        Only labels that appear in labels_to_keep are used. Their index
        in labels_to_keep are used for labels. The default label is 0,
        thus labels_to_keep should start with "none" """
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_cam = T_lidar_to_cam @ pts_h.T
    pts_2d = P_cam @ pts_cam
    pts_2d /= pts_2d[2, :]
    pts_2d = np.round(pts_2d).T.astype(np.int)

    # maybe change with pts_2d = pts_2d[pts_2d[:, 0]<w]
#    valid0 = np.where(pts_2d[:, 0] < w)[0]
#    valid1 = np.where(pts_2d[:, 0] >= 0)[0]
#    valid2 = np.where(pts_2d[:, 1] < h)[0]
#    valid3 = np.where(pts_2d[:, 1] >= 0)[0]
#    valid4 = np.where(pts_cam.T[:, 2] > 0)[0]
#    valid_idx = np.intersect1d(np.intersect1d(np.intersect1d(valid0, valid1),
#                                              np.intersect1d(valid2, valid3)),
#                               valid4)
    pts_cam = pts_cam.T
    filter_in_front = np.where(pts_cam[:, 2] > 0)[0]
    # default label is 0
    labels = np.zeros((pts.shape[0]), dtype=np.uint8)
    for cls in detections_2d:
        if cls in labels_to_keep:
            v = labels_to_keep.index(cls)
            for x0, y0, x1, y1 in detections_2d[cls]:
                i0 = np.where(pts_2d[:, 0] >= x0)[0]
                i1 = np.where(pts_2d[:, 0] <= x1)[0]
                i2 = np.where(pts_2d[:, 1] >= y0)[0]
                i3 = np.where(pts_2d[:, 1] <= y1)[0]
                inside_bb_idx = np.intersect1d(np.intersect1d(
                                    np.intersect1d(i0, i1),
                                    np.intersect1d(i2, i3)),
                                filter_in_front)
                d = pts_cam[inside_bb_idx, 2]
                d_ref = np.percentile(d, 25)
                filt = np.where(abs(pts_cam[:, 2]-d_ref) < 2)[0]
                good_points = np.intersect1d(filt, inside_bb_idx)
                labels[good_points] = v
    return labels


T_lidar_to_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                           [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                           [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                           [0, 0, 0, 1]])

# general cam to image_02_rect
P_cam = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                  [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])


if __name__ == '__main__':
    # example
    working_dir = "/home/matthieu/Downloads/2011_09_26/2011_09_26_drive_0035_sync/"

    # images
    images_list = glob.glob(os.path.join(working_dir, "image_02/data/*.png"))
    images_names = []
    for f in images_list:
        images_names.append(os.path.splitext(os.path.basename(f))[0])
    images_names = sorted(images_names)

    # labels settings
    labels_to_keep = ["none", "car", "person"]
    labels_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    detections_filename = "/home/matthieu/Lib/kwiver/build-sprokit/examples/pipelines/darknet/output/images/detections.csv"

    for name in images_names:
        print(name)
        # load pointcloud
        pts = np.fromfile(os.path.join(working_dir, "velodyne_points/data/", name + ".bin"), np.float32, -1)
        pts = pts.reshape((-1, 4))
        pts = pts[:, :3]
        # load image
        img = cv2.imread(os.path.join(working_dir, "image_02/data/", name + ".png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load detections
        detections_2d = load_detections_2d(detections_filename, name)

        labels = assign_labels(pts, T_lidar_to_cam, P_cam, img.shape[1],
                               img.shape[0], detections_2d, labels_to_keep)

        out_dir_obj = "obj/"
        with open(os.path.join(working_dir, out_dir_obj, "pc_" + name + ".obj"), "w") as fout:
            for i, p in enumerate(pts):
                fout.write("v " + " ".join(p.astype(str)) + " " +
                           " ".join(map(str, labels_colors[labels[i]])) + "\n")

#    # Write pointcloud with a color per class
#    header = "ply\n" \
#             "format ascii 1.0\n" \ labels_to_keep)
#
#             "comment VCGLIB generated\n" \
#             "element vertex " + str(pts.shape[0]) + "\n" \
#             "property float x\n" \
#             "property float y\n" \
#             "property float z\n" \
#             "property uchar red\n" \
#             "property uchar green\n" \
#             "property uchar blue\n" \
#             "element face 0\n" \
#             "property list uchar int vertex_indices\n" \
#             "end_header\n"
#    out_dir_ply = "ply/"
#    with open(out_dir_ply + "pc_" + name + ".ply", "w") as fout:
#        fout.write(header)
#        for i, p in enumerate(pts):
#            fout.write(" ".join(p.astype(str)) + " " +
#                       " ".join(map(str, labels_colors[labels[i]])) + "\n")
